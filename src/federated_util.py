from copy import deepcopy
from types import SimpleNamespace
from typing import List, Set, Tuple, Callable, Optional

import numpy as np
import torch
from context_printer import ContextPrinter as Ctp, Color
from torch.utils.data import DataLoader

from architectures import NormalizingModel
from ml import set_models_sub_divs


# ------------------------------------------------------------
# Standard Federated Averaging
# ------------------------------------------------------------
def federated_averaging(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    with torch.no_grad():
        state_dict_mean = global_model.state_dict()
        for key in state_dict_mean:
            state_dict_mean[key] = torch.stack(
                [model.state_dict()[key] for model in models], dim=-1
            ).mean(dim=-1)
        global_model.load_state_dict(state_dict_mean)


# ------------------------------------------------------------
# Differential Privacy Federated Averaging (DP-FedAvg)
# ------------------------------------------------------------
def federated_averaging_dp(global_model: torch.nn.Module,
                           models: List[torch.nn.Module],
                           noise_multiplier: float,
                           clip_norm: float) -> None:
    """
    Implements Federated Averaging with Differential Privacy
    """
    with torch.no_grad():
        global_state = global_model.state_dict()
        client_states = [model.state_dict() for model in models]
        num_clients = len(models)

        new_global_state = {}

        for key in global_state:
            updates = []
            for client_state in client_states:
                update = client_state[key] - global_state[key]

                norm = torch.norm(update, p=2)
                scale = min(1.0, clip_norm / (norm + 1e-6))
                updates.append(update * scale)

            stacked_updates = torch.stack(updates, dim=0)
            avg_update = torch.mean(stacked_updates, dim=0)

            noise_std = (clip_norm * noise_multiplier) / num_clients
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=avg_update.shape
            ).to(avg_update.device)

            new_global_state[key] = global_state[key] + avg_update + noise

        global_model.load_state_dict(new_global_state)


# ------------------------------------------------------------
# Robust Aggregations
# ------------------------------------------------------------
def federated_median(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    with torch.no_grad():
        state_dict_median = global_model.state_dict()
        for key in state_dict_median:
            sorted_tensor, _ = torch.sort(
                torch.stack([model.state_dict()[key] for model in models], dim=-1),
                dim=-1
            )
            state_dict_median[key] = sorted_tensor.median(dim=-1)[0]
        global_model.load_state_dict(state_dict_median)


def federated_trimmed_mean(global_model: torch.nn.Module,
                           models: List[torch.nn.Module],
                           trim: int) -> None:
    with torch.no_grad():
        state_dict = global_model.state_dict()
        for key in state_dict:
            sorted_tensor, _ = torch.sort(
                torch.stack([model.state_dict()[key] for model in models], dim=-1),
                dim=-1
            )
            trimmed = sorted_tensor[..., trim:-trim]
            state_dict[key] = trimmed.mean(dim=-1)
        global_model.load_state_dict(state_dict)


# ------------------------------------------------------------
# Poisoning Attacks
# ------------------------------------------------------------
def model_update_scaling(global_model: torch.nn.Module,
                         malicious_clients_models: List[torch.nn.Module],
                         factor: float) -> None:
    with torch.no_grad():
        for model in malicious_clients_models:
            new_state = {}
            for key, original in global_model.state_dict().items():
                delta = model.state_dict()[key] - original
                new_state[key] = original + delta * factor
            model.load_state_dict(new_state)


def mimic_attack(models: List[torch.nn.Module],
                 malicious_clients: Set[int],
                 mimicked_client: int) -> None:
    with torch.no_grad():
        for i in malicious_clients:
            models[i].load_state_dict(models[mimicked_client].state_dict())


# ------------------------------------------------------------
# Model Poisoning Controller
# ------------------------------------------------------------
def model_poisoning(global_model: torch.nn.Module,
                    models: List[torch.nn.Module],
                    params: SimpleNamespace,
                    mimicked_client_id: Optional[int] = None,
                    verbose: bool = False):

    malicious = [m for i, m in enumerate(models) if i in params.malicious_clients]

    if params.model_poisoning == "mimic_attack":
        mimic_attack(models, params.malicious_clients, mimicked_client_id)
        if verbose:
            Ctp.print("Mimic attack executed")

    model_update_scaling(global_model, malicious, params.model_update_factor)

    return models


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def init_federated_models(train_dls: List[DataLoader],
                          params: SimpleNamespace,
                          architecture: Callable):

    n_clients = len(params.clients_devices)

    global_model = NormalizingModel(
        architecture(
            activation_function=params.activation_fn,
            hidden_layers=params.hidden_layers
        ),
        sub=torch.zeros(params.n_features),
        div=torch.ones(params.n_features)
    )

    if params.cuda:
        global_model = global_model.cuda()

    models = [deepcopy(global_model) for _ in range(n_clients)]

    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    federated_averaging(global_model, models)

    models = [deepcopy(global_model) for _ in range(n_clients)]

    return global_model, models


# ------------------------------------------------------------
# FINAL AGGREGATION FUNCTION (WITH DP SUPPORT)
# ------------------------------------------------------------
def model_aggregation(global_model: torch.nn.Module,
                      models: List[torch.nn.Module],
                      params: SimpleNamespace,
                      verbose: bool = False) -> Tuple[torch.nn.Module, List[torch.nn.Module]]:

    if hasattr(params, "use_dp") and params.use_dp:
        noise_multiplier = getattr(params, "noise_multiplier", 1.0)
        clip_norm = getattr(params, "clip_norm", 1.0)

        if verbose:
            print("Using Differential Privacy Aggregation")

        federated_averaging_dp(global_model, models, noise_multiplier, clip_norm)

    else:
        params.aggregation_function(global_model, models)

    models = [deepcopy(global_model) for _ in range(len(models))]

    return global_model, models


def select_mimicked_client(params):
    honest_client_ids = [
        client_id for client_id in range(len(params.clients_devices))
        if client_id not in params.malicious_clients
    ]

    if params.model_poisoning == 'mimic_attack':
        mimicked_client_id = np.random.choice(honest_client_ids)
        print('The mimicked client is {}'.format(mimicked_client_id))
    else:
        mimicked_client_id = None

    return mimicked_client_id
