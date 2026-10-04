from argparse import ArgumentParser
import torch
import torch.utils.data

from data import read_all_data, all_devices
from federated_util import *
from grid_search import run_grid_search
from test_hparams import test_hyperparameters
from unsupervised_data import get_client_unsupervised_initial_splitting
from architectures import VAE


# =========================================================
# SELECT EXPERIMENT TYPE HERE
# Options: "clean", "attack", "dp_clean", "dp_attack"
# =========================================================
EXPERIMENT_TYPE = "dp_attack"


def main(setup: str, experiment: str, federated: str, test: bool, collaborative: bool):
    Ctp.set_automatic_skip(True)

    # =========================================================
    # EXPERIMENT SWITCH
    # =========================================================
    if EXPERIMENT_TYPE == "clean":
        model_poisoning = None
        n_malicious = 0
        malicious_clients = []
        model_update_factor = 1.0
        use_dp = False
        noise_multiplier = 0.0
        clip_norm = 1.0
        run_name = "run_clean"

    elif EXPERIMENT_TYPE == "attack":
        model_poisoning = "model_cancelling"
        n_malicious = 1
        malicious_clients = [7]
        model_update_factor = -7.0
        use_dp = False
        noise_multiplier = 0.0
        clip_norm = 1.0
        run_name = "run_attack"

    elif EXPERIMENT_TYPE == "dp_clean":
        model_poisoning = None
        n_malicious = 0
        malicious_clients = []
        model_update_factor = 1.0
        use_dp = True
        noise_multiplier = 0.3
        clip_norm = 1.0
        run_name = "run_dp_clean"

    elif EXPERIMENT_TYPE == "dp_attack":
        model_poisoning = "model_cancelling"
        n_malicious = 1
        malicious_clients = [7]
        model_update_factor = -7.0
        use_dp = True
        noise_multiplier = 0.3
        clip_norm = 1.0
        run_name = "run_dp_attack"

    else:
        raise ValueError("Invalid EXPERIMENT_TYPE. Choose from: clean, attack, dp_clean, dp_attack")

    # =========================================================
    # COMMON PARAMETERS
    # =========================================================
    common_params = {
        'n_features': 115,
        'normalization': 'min-max',
        'test_bs': 4096,
        'p_test': 0.2,
        'p_unused': 0.01,
        'val_part': None,
        'n_splits': 5,
        'n_random_reruns': 1,
        'cuda': torch.cuda.is_available(),
        'benign_prop': 0.0787,
        'samples_per_device': 100_000,

        # Security / Attack
        'model_poisoning': model_poisoning,
        'n_malicious': n_malicious,
        'model_update_factor': model_update_factor,
        'malicious_clients': malicious_clients,

        # Differential Privacy
        'use_dp': use_dp,
        'noise_multiplier': noise_multiplier,
        'clip_norm': clip_norm,

        # Save name
        'run_name': run_name,
    }

    # =========================================================
    # SPLIT CALCULATION
    # =========================================================
    p_train_val = 1.0 - common_params['p_test'] - common_params['p_unused']
    val_part = 1.0 / common_params['n_splits'] if common_params['val_part'] is None else common_params['val_part']

    common_params.update({
        'p_train_val': p_train_val,
        'val_part': val_part
    })

    # =========================================================
    # AUTOENCODER / VAE PARAMETERS
    # =========================================================
    autoencoder_params = {
        'activation_fn': torch.nn.ELU,
        'threshold_part': 0.5,
        'quantile': 0.95,

        # Training
        'epochs': 120,
        'train_bs': 64,

        # Optimizer
        'optimizer': torch.optim.Adam,
        'optimizer_params': {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        },

        # Scheduler
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_params': {
            'step_size': 10,
            'gamma': 0.5,
        },

        # Proposed architecture
        'architecture': VAE,
    }

    # =========================================================
    # FEDERATION PARAMETERS
    # =========================================================
    federation_params = {
        'aggregation_function': federated_averaging,
        'resampling': None,
    }

    if federated == 'fedavg':
        federation_params.update({
            'federation_rounds': 30,
            'gamma_round': 0.75,
        })

    elif federated == 'fedsgd':
        federation_params.update({
            'train_bs': 8
        })

    # =========================================================
    # DATA LOADING
    # =========================================================
    all_data = read_all_data()
    n_devices = len(all_devices)

    # =========================================================
    # CONFIGURATIONS
    # =========================================================
    centralized_configurations = [
        {
            'clients_devices': [[i for i in range(n_devices) if i != test_device]],
            'test_devices': [test_device],
        }
        for test_device in range(n_devices)
    ]

    decentralized_configurations = [
        {
            'clients_devices': [[i] for i in range(n_devices) if i != test_device],
            'test_devices': [test_device],
        }
        for test_device in range(n_devices)
    ]

    if setup == 'centralized':
        configurations = centralized_configurations

    elif setup == 'decentralized':
        configurations = decentralized_configurations

    else:
        raise ValueError("Setup must be 'centralized' or 'decentralized'.")

    if experiment != 'autoencoder':
        raise ValueError("Only 'autoencoder' supported.")

    constant_params = {**common_params, **autoencoder_params}

    if federated is not None:
        constant_params.update(federation_params)

    splitting_function = get_client_unsupervised_initial_splitting

    # =========================================================
    # TEST / FINAL RUN MODE
    # =========================================================
    if test:
        print("\n========================================")
        print(f"🚀 Running Experiment: {EXPERIMENT_TYPE.upper()}")
        print("========================================")

        if EXPERIMENT_TYPE == "clean":
            print("Mode: Clean / No DP / No Attack")
        elif EXPERIMENT_TYPE == "attack":
            print("Mode: Model Cancelling Attack / No DP")
        elif EXPERIMENT_TYPE == "dp_clean":
            print("Mode: DP Clean / DP Enabled / No Attack")
        elif EXPERIMENT_TYPE == "dp_attack":
            print("Mode: DP + Model Cancelling Attack")

        print(f"Federated rounds: {federation_params.get('federation_rounds', 'N/A')}")
        print(f"Local epochs: {autoencoder_params['epochs']}")
        print("Architecture: VAE")
        print(f"DP Enabled: {use_dp}")
        print(f"Noise multiplier: {noise_multiplier}")
        print(f"Clip norm: {clip_norm}")
        print(f"Model poisoning: {model_poisoning}")
        print(f"Malicious clients: {malicious_clients}")
        print(f"Model update factor: {model_update_factor}")
        print("========================================\n")

        configurations_params = [
            {
                'hidden_layers': [29],
                'optimizer_params': {
                    'lr': 1e-4,
                    'weight_decay': 1e-4,
                },
            }
            for _ in range(len(configurations))
        ]

        test_hyperparameters(
            all_data,
            setup,
            experiment,
            federated,
            splitting_function,
            constant_params,
            configurations_params,
            configurations,
        )

        print("\n==============================")
        print("Final Results")
        print("==============================")
        print(f"{EXPERIMENT_TYPE.upper()} Training Finished ✅")
        print("==============================")
        return

    # =========================================================
    # GRID SEARCH MODE
    # =========================================================
    print(f"\n🚀 Starting Grid Search for {EXPERIMENT_TYPE.upper()}...\n")

    varying_params = {
        'hidden_layers': [[29]],
        'optimizer_params': [
            {
                'lr': 1e-4,
                'weight_decay': 1e-4,
            },
            {
                'lr': 5e-5,
                'weight_decay': 1e-4,
            },
        ],
    }

    run_grid_search(
        all_data,
        setup,
        experiment,
        splitting_function,
        constant_params,
        varying_params,
        configurations,
        collaborative,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('setup')
    parser.add_argument('experiment')

    parser.add_argument(
        '--fedavg',
        dest='federated',
        action='store_const',
        const='fedavg'
    )

    parser.add_argument(
        '--fedsgd',
        dest='federated',
        action='store_const',
        const='fedsgd'
    )

    parser.set_defaults(federated=None)

    parser.add_argument(
        '--test',
        dest='test',
        action='store_true'
    )

    parser.set_defaults(test=False)

    parser.add_argument(
        '--collaborative',
        dest='collaborative',
        action='store_true'
    )

    parser.set_defaults(collaborative=True)

    args = parser.parse_args()

    main(
        args.setup,
        args.experiment,
        args.federated,
        args.test,
        args.collaborative
    )