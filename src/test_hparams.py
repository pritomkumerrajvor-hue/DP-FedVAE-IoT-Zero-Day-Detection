from copy import deepcopy
from time import time
from types import SimpleNamespace
from typing import Callable, Tuple, List, Dict, Optional

import numpy as np
from context_printer import ContextPrinter as Ctp, Color

from data import FederationData, ClientData, DeviceData, get_configuration_data, get_initial_splitting
from metrics import BinaryClassificationResult
from saving import create_new_numbered_dir, save_results_test

# ✅ Supervised
from supervised_experiments import (
    local_classifiers_train_test,
    fedavg_classifiers_train_test,
    fedsgd_classifiers_train_test
)

# ✅ Unsupervised
from unsupervised_experiments import (
    local_autoencoders_train_test,
    fedavg_autoencoders_train_test
)


# ---------------------------------------------------------
# Select experiment function safely
# ---------------------------------------------------------

def select_experiment_function(experiment: str, federated: Optional[str]) -> Callable:

    if federated is not None:

        if federated == 'fedavg':
            if experiment == 'classifier':
                return fedavg_classifiers_train_test
            elif experiment == 'autoencoder':
                return fedavg_autoencoders_train_test

        elif federated == 'fedsgd':
            if experiment == 'classifier':
                return fedsgd_classifiers_train_test
            else:
                raise ValueError("FedSGD not implemented for autoencoder.")

        else:
            raise ValueError("Unknown federated method.")

    else:
        if experiment == 'classifier':
            return local_classifiers_train_test
        elif experiment == 'autoencoder':
            return local_autoencoders_train_test

    raise ValueError("Invalid experiment type.")


# ---------------------------------------------------------
# Print metrics safely
# ---------------------------------------------------------

def print_metrics_block(title: str, res) -> None:
    print(f"\n==============================")
    print(f"{title}")
    print(f"==============================")

    # raw object / fields
    try:
        if hasattr(res, '__dict__'):
            print("Available fields:")
            for k, v in vars(res).items():
                print(f"  {k}: {v}")
        else:
            print("Raw result:", res)
    except Exception as e:
        print("Could not print raw result fields:", e)

    # common metric names
    metric_fields = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'f1_score',
        'tp',
        'tn',
        'fp',
        'fn',
        'confusion_matrix'
    ]

    print("\nCommon metrics:")
    found_any = False
    for attr in metric_fields:
        if hasattr(res, attr):
            found_any = True
            print(f"{attr}: {getattr(res, attr)}")

    if not found_any:
        print("No standard metric fields found in this result object.")


# ---------------------------------------------------------
# Multiple reruns for confidence
# ---------------------------------------------------------

def compute_rerun_results(
        clients_train_val: FederationData,
        clients_test: FederationData,
        test_devices_data: ClientData,
        experiment: str,
        federated: Optional[str],
        params: SimpleNamespace
) -> Tuple[List[BinaryClassificationResult],
           List[BinaryClassificationResult],
           Optional[List[List[float]]]]:

    local_results = []
    new_devices_results = []
    thresholds = []

    experiment_function = select_experiment_function(experiment, federated)

    for run_id in range(params.n_random_reruns):

        Ctp.enter_section(
            f'Run [{run_id + 1}/{params.n_random_reruns}]',
            Color.GRAY
        )

        if federated is not None and hasattr(params, "n_malicious"):
            malicious_clients = set(
                np.random.choice(len(clients_train_val),
                                 params.n_malicious,
                                 replace=False)
            )
            params.malicious_clients = malicious_clients
            Ctp.print('Malicious clients: ' + repr(list(malicious_clients)))

        start_time = time()

        result = experiment_function(
            clients_train_val,
            clients_test,
            test_devices_data,
            params=params
        )

        local_res = result[0]
        new_res = result[1]

        print_metrics_block("Local Result", local_res)
        print_metrics_block("New Device Result", new_res)

        local_results.append(local_res)
        new_devices_results.append(new_res)

        threshold = result[2] if experiment == 'autoencoder' else None

        if threshold is not None:
            thresholds.append(threshold)
            print("\nThreshold:", threshold)

        Ctp.print(f"Elapsed time: {time() - start_time:.1f} seconds")
        Ctp.exit_section()

    return local_results, new_devices_results, thresholds


# ---------------------------------------------------------
# Test Hyperparameters
# ---------------------------------------------------------

def test_hyperparameters(
        all_data: List[DeviceData],
        setup: str,
        experiment: str,
        federated: Optional[str],
        splitting_function: Callable,
        constant_params: dict,
        configurations_params: List[dict],
        configurations: List[Dict[str, list]]
) -> None:

    base_path = (
        'test_results/'
        + setup + '_'
        + experiment
        + ('_' + federated if federated is not None else '')
        + '/run_'
    )

    params_dict = deepcopy(constant_params)

    local_results = {}
    new_devices_results = {}
    thresholds = {}

    for j, (configuration, configuration_params) in enumerate(
            zip(configurations, configurations_params)):

        clients_devices_data, test_devices_data = get_configuration_data(
            all_data,
            configuration['clients_devices'],
            configuration['test_devices']
        )

        clients_train_val, clients_test = get_initial_splitting(
            splitting_function,
            clients_devices_data,
            p_test=params_dict['p_test'],
            p_unused=params_dict['p_unused']
        )

        params_dict.update(configuration)
        params_dict.update(configuration_params)

        params = SimpleNamespace(**params_dict)

        Ctp.enter_section(
            f'Configuration [{j + 1}/{len(configurations)}]: {configuration}',
            Color.NONE
        )

        local_result, new_result, threshold = compute_rerun_results(
            clients_train_val,
            clients_test,
            test_devices_data,
            experiment,
            federated,
            params
        )

        local_results[repr(configuration)] = local_result
        new_devices_results[repr(configuration)] = new_result
        thresholds[repr(configuration)] = threshold

        Ctp.exit_section()

    if experiment != 'autoencoder':
        thresholds = None

    results_path = create_new_numbered_dir(base_path)

    save_results_test(
        results_path,
        local_results,
        new_devices_results,
        thresholds,
        constant_params,
        configurations_params
    )