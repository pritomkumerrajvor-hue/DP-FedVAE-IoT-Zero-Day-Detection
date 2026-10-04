from typing import Tuple, Dict, List, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from context_printer import Color
from context_printer import ContextPrinter as Ctp
from sklearn.model_selection import KFold


all_devices = [
    'Danmini_Doorbell',
    'Ecobee_Thermostat',
    'Ennio_Doorbell',
    'Philips_B120N10_Baby_Monitor',
    'Provision_PT_737E_Security_Camera',
    'Provision_PT_838_Security_Camera',
    'Samsung_SNH_1011_N_Webcam',
    'SimpleHome_XCS7_1002_WHT_Security_Camera',
    'SimpleHome_XCS7_1003_WHT_Security_Camera'
]


# List of the devices that can be infected by the Mirai malware
mirai_devices = all_devices[0:2] + all_devices[3:6] + all_devices[7:9]

mirai_attacks = ['ack', 'scan', 'syn', 'udp', 'udpplain']
gafgyt_attacks = ['combo', 'junk', 'scan', 'tcp', 'udp']


# ---------------------------------------------------------
# Dataset path
# ---------------------------------------------------------

# data.py is inside: project_root/src/data.py
# Therefore parent.parent gives the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "data"
    / "N-BaIoT"
    / "detection+of+iot+botnet+attacks+n+baiot"
)


# Benign traffic paths
benign_paths = {
    device: data_path / device / "benign_traffic.csv"
    for device in all_devices
}


# Mirai attack paths
mirai_paths = [
    {
        device: data_path / device / "mirai_attacks" / f"{attack}.csv"
        for device in mirai_devices
    }
    for attack in mirai_attacks
]


# Gafgyt attack paths
gafgyt_paths = [
    {
        device: data_path / device / "gafgyt_attacks" / f"{attack}.csv"
        for device in all_devices
    }
    for attack in gafgyt_attacks
]


multiclass_labels = {
    **{'benign': 0.},
    **{'mirai_' + attack: float(i + 1) for i, attack in enumerate(mirai_attacks)},
    **{'gafgyt_' + attack: float(i + 6) for i, attack in enumerate(gafgyt_attacks)}
}