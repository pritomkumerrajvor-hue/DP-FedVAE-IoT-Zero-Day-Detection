from types import SimpleNamespace
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader

from context_printer import Color
from context_printer import ContextPrinter as Ctp

from architectures import SimpleAutoencoder, NormalizingModel, Threshold
from data import (
    device_names,
    ClientData,
    FederationData,
    get_benign_attack_samples_per_device,
)
from federated_util import (
    init_federated_models,
    model_aggregation,
    select_mimicked_client,
    model_poisoning,
)
from metrics import BinaryClassificationResult
from ml import set_models_sub_divs, set_model_sub_div
from print_util import print_federation_round
from unsupervised_data import get_train_dl, get_val_dl, prepare_dataloaders

# ✅ FIXED IMPORT
from unsupervised_ml import (
    multitrain_autoencoders,
    multitest_autoencoders,
    compute_thresholds,
    train_autoencoder,
    compute_reconstruction_losses,
    train_autoencoders_fedsgd,
    federated_thresholds,
    federated_testing,
)


# =========================================================
# 1️⃣ Local Autoencoder Training + Validation
# =========================================================
def local_autoencoder_train_val(
    train_data: ClientData,
    val_data: ClientData,
    params: SimpleNamespace,
) -> float:

    model_class = getattr(params, "architecture", SimpleAutoencoder)

    p_train = params.p_train_val * (1.0 - params.val_part)
    p_val = params.p_train_val * params.val_part

    benign_samples_per_device, _ = get_benign_attack_samples_per_device(
        p_split=p_train,
        benign_prop=1.0,
        samples_per_device=params.samples_per_device,
    )

    train_dl = get_train_dl(
        train_data,
        params.train_bs,
        benign_samples_per_device=benign_samples_per_device,
        cuda=params.cuda,
    )

    benign_samples_per_device, _ = get_benign_attack_samples_per_device(
        p_split=p_val,
        benign_prop=1.0,
        samples_per_device=params.samples_per_device,
    )

    val_dl = get_val_dl(
        val_data,
        params.test_bs,
        benign_samples_per_device=benign_samples_per_device,
        cuda=params.cuda,
    )

    model = NormalizingModel(
        model_class(
            activation_function=params.activation_fn,
            hidden_layers=params.hidden_layers,
        ),
        sub=torch.zeros(params.n_features),
        div=torch.ones(params.n_features),
    )

    if params.cuda:
        model = model.cuda()

    set_model_sub_div(params.normalization, model, train_dl)

    Ctp.enter_section(
        f"Training for {params.epochs} epochs with {len(train_dl.dataset[:][0])} samples",
        color=Color.GREEN,
    )

    train_autoencoder(model, params, train_dl)
    Ctp.exit_section()

    Ctp.print(f"Validating with {len(val_dl.dataset[:][0])} samples")
    losses = compute_reconstruction_losses(model, val_dl)
    loss = (sum(losses) / len(losses)).item()

    Ctp.print(f"Validation loss: {loss:.5f}")
    return loss


# =========================================================
# 2️⃣ Local Multi-Client Training & Testing
# =========================================================
def local_autoencoders_train_test(
    train_val_data: FederationData,
    local_test_data: FederationData,
    new_test_data: ClientData,
    params: SimpleNamespace,
) -> Tuple[BinaryClassificationResult, BinaryClassificationResult, List[float]]:

    model_class = getattr(params, "architecture", SimpleAutoencoder)

    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(
        train_val_data,
        local_test_data,
        new_test_data,
        params,
    )

    n_clients = len(params.clients_devices)

    models = [
        NormalizingModel(
            model_class(
                activation_function=params.activation_fn,
                hidden_layers=params.hidden_layers,
            ),
            sub=torch.zeros(params.n_features),
            div=torch.ones(params.n_features),
        )
        for _ in range(n_clients)
    ]

    if params.cuda:
        models = [model.cuda() for model in models]

    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    multitrain_autoencoders(
        trains=list(
            zip(
                [
                    f"Training client {i} on: {device_names(client_devices)}"
                    for i, client_devices in enumerate(params.clients_devices)
                ],
                train_dls,
                models,
            )
        ),
        params=params,
        main_title="Training the clients",
        color=Color.GREEN,
    )

    thresholds = compute_thresholds(
        opts=list(
            zip(
                [
                    f"Computing threshold for client {i} on: {device_names(client_devices)}"
                    for i, client_devices in enumerate(params.clients_devices)
                ],
                threshold_dls,
                models,
            )
        ),
        quantile=params.quantile,
        main_title="Computing thresholds",
        color=Color.DARK_PURPLE,
    )

    local_result = multitest_autoencoders(
        tests=list(
            zip(
                [
                    f"Testing client {i} on: {device_names(client_devices)}"
                    for i, client_devices in enumerate(params.clients_devices)
                ],
                local_test_dls_dicts,
                models,
                thresholds,
            )
        ),
        main_title="Testing clients on their own devices",
        color=Color.BLUE,
    )

    new_devices_result = multitest_autoencoders(
        tests=list(
            zip(
                [
                    f"Testing client {i} on: {device_names(params.test_devices)}"
                    for i in range(n_clients)
                ],
                [new_test_dls_dict for _ in range(n_clients)],
                models,
                thresholds,
            )
        ),
        main_title="Testing clients on new devices",
        color=Color.DARK_CYAN,
    )

    return (
        local_result,
        new_devices_result,
        [t.threshold.item() for t in thresholds],
    )


# =========================================================
# 3️⃣ Federated (FedAvg)
# =========================================================
def fedavg_autoencoders_train_test(
    train_val_data: FederationData,
    local_test_data: FederationData,
    new_test_data: ClientData,
    params: SimpleNamespace,
) -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult], List[float]]:

    model_class = getattr(params, "architecture", SimpleAutoencoder)

    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(
        train_val_data,
        local_test_data,
        new_test_data,
        params,
    )

    global_model, models = init_federated_models(
        train_dls,
        params,
        architecture=model_class,
    )

    global_threshold = Threshold(torch.tensor(0.0))

    local_results = []
    new_devices_results = []
    global_thresholds = []

    mimicked_client_id = select_mimicked_client(params)

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        multitrain_autoencoders(
            trains=list(
                zip(
                    [
                        f"Training client {i} on: {device_names(client_devices)}"
                        for i, client_devices in enumerate(params.clients_devices)
                    ],
                    train_dls,
                    models,
                )
            ),
            params=params,
            lr_factor=(params.gamma_round ** federation_round),
            main_title="Training clients",
            color=Color.GREEN,
        )

        models = model_poisoning(
            global_model,
            models,
            params,
            mimicked_client_id=mimicked_client_id,
            verbose=True,
        )

        global_model, models = model_aggregation(
            global_model,
            models,
            params,
            verbose=True,
        )

        federated_thresholds(
            models,
            threshold_dls,
            global_threshold,
            params,
            global_thresholds,
        )

        federated_testing(
            global_model,
            global_threshold,
            local_test_dls_dicts,
            new_test_dls_dict,
            params,
            local_results,
            new_devices_results,
        )

        Ctp.exit_section()

    return local_results, new_devices_results, global_thresholds
