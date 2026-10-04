import itertools
from copy import deepcopy
from time import time
from types import SimpleNamespace
from typing import List, Dict, Callable, Union

from context_printer import ContextPrinter as Ctp, Color

from data import (
    ClientData,
    split_client_data_current_fold,
    split_client_data,
    DeviceData,
    device_names,
    get_client_data
)
from metrics import BinaryClassificationResult
from saving import create_new_numbered_dir, save_results_gs
from supervised_experiments import local_classifier_train_val
from unsupervised_experiments import local_autoencoder_train_val


# --------------------------------------------
# Helper: Get all unique client-device groups
# --------------------------------------------
def get_all_clients_devices(configurations: List[Dict[str, list]]) -> List[tuple]:
    all_clients_devices_dict = {}

    for configuration in reversed(configurations):
        for client_devices in configuration['clients_devices']:
            all_clients_devices_dict[tuple(client_devices)] = 0

    return list(all_clients_devices_dict.keys())


# --------------------------------------------
# Cross-validation evaluation
# --------------------------------------------
def compute_cv_result(train_val_data: ClientData,
                      experiment: str,
                      params: SimpleNamespace,
                      n_splits: int) -> Union[BinaryClassificationResult, float]:

    result = BinaryClassificationResult() if experiment == 'classifier' else 0.0

    for fold in range(n_splits):
        Ctp.enter_section(f'Fold [{fold + 1}/{n_splits}]', Color.GRAY)

        train_data, val_data = split_client_data_current_fold(
            train_val_data, n_splits, fold
        )

        if experiment == 'classifier':
            result += local_classifier_train_val(train_data, val_data, params=params)

        elif experiment == 'autoencoder':
            result += local_autoencoder_train_val(train_data, val_data, params=params)

        else:
            raise ValueError("Unknown experiment type")

        Ctp.exit_section()

    return result


# --------------------------------------------
# Single split evaluation
# --------------------------------------------
def compute_single_split_result(train_val_data: ClientData,
                                experiment: str,
                                params: SimpleNamespace,
                                val_part: float) -> Union[BinaryClassificationResult, float]:

    train_data, val_data = split_client_data(
        train_val_data,
        p_second_split=val_part,
        p_unused=0.0
    )

    if experiment == 'classifier':
        return local_classifier_train_val(train_data, val_data, params=params)

    elif experiment == 'autoencoder':
        return local_autoencoder_train_val(train_data, val_data, params=params)

    else:
        raise ValueError("Unknown experiment type")


# --------------------------------------------
# Main Grid Search
# --------------------------------------------
def run_grid_search(all_data: List[DeviceData],
                    setup: str,
                    experiment: str,
                    splitting_function: Callable,
                    constant_params: dict,
                    varying_params: dict,
                    configurations: List[Dict[str, list]],
                    collaborative: bool = False) -> None:

    base_path = f'grid_search_results/{setup}_{experiment}/run_'

    params_product = list(itertools.product(*varying_params.values()))
    base_params = deepcopy(constant_params)

    if base_params['n_splits'] == 1 and base_params['val_part'] is None:
        raise ValueError('val_part must be specified when not using cross-validation')

    all_clients_devices = get_all_clients_devices(configurations)
    Ctp.print(all_clients_devices)

    clients_results = {}

    for i, client_devices_tuple in enumerate(all_clients_devices):

        client_devices = list(client_devices_tuple)

        Ctp.enter_section(
            f'Client {i} with devices: {device_names(client_devices)}',
            Color.WHITE
        )

        client_data = get_client_data(all_data, client_devices)

        train_val_data, _ = splitting_function(
            client_data,
            p_test=base_params['p_test'],
            p_unused=base_params['p_unused']
        )

        clients_results[repr(client_devices)] = {}

        for j, params_tuple in enumerate(params_product):

            start_time = time()

            experiment_params = dict(
                zip(varying_params.keys(), params_tuple)
            )

            # 🔥 IMPORTANT FIX: fresh copy per experiment
            current_params = deepcopy(base_params)
            current_params.update(experiment_params)

            Ctp.enter_section(
                f'Experiment [{j + 1}/{len(params_product)}] with params: {experiment_params}',
                Color.NONE
            )

            params = SimpleNamespace(**current_params)

            if current_params['n_splits'] == 1:
                result = compute_single_split_result(
                    train_val_data,
                    experiment,
                    params,
                    current_params['val_part']
                )
            else:
                result = compute_cv_result(
                    train_val_data,
                    experiment,
                    params,
                    current_params['n_splits']
                )

            clients_results[repr(client_devices)][repr(experiment_params)] = result

            Ctp.print(f"Elapsed time: {time() - start_time:.1f} seconds")
            Ctp.exit_section()

        Ctp.exit_section()

    # --------------------------------------------
    # Save results
    # --------------------------------------------
    results_path = create_new_numbered_dir(base_path)

    if collaborative:

        configurations_results = {}

        for configuration in configurations:

            config_key = repr(configuration['clients_devices'])
            configurations_results[config_key] = {}

            for params_tuple in params_product:

                experiment_params = dict(
                    zip(varying_params.keys(), params_tuple)
                )

                total_result = BinaryClassificationResult() \
                    if experiment == 'classifier' else 0.0

                for client_devices in configuration['clients_devices']:
                    total_result += clients_results[
                        repr(client_devices)
                    ][repr(experiment_params)]

                configurations_results[config_key][repr(experiment_params)] = total_result

        save_results_gs(results_path, configurations_results, constant_params)

    else:
        save_results_gs(results_path, clients_results, constant_params)
