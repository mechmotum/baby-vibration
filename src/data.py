import os
import collections
from datetime import datetime

import yaml
import pandas as pd

with open('config.yml') as f:
    config_data = yaml.safe_load(f)

PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
PATH_TO_DATA = config_data['data-directory']
PATH_TO_FILE = os.path.join(
    PATH_TO_DATA,
    'Raw_data_csv',
    '2024-11-28_10.43.58_maxicos_1_SH_SD_Session1',
    'maxicos_1_SH_Session1_S_SeatHead_Calibrated_SD.csv')

"""
First three lines look like this:

"sep=,"
S_SeatHead_Timestamp_Unix_CAL,S_SeatHead_Accel_WR_X_CAL,S_SeatHead_Accel_WR_Y_CAL,S_SeatHead_Accel_WR_Z_CAL,S_SeatHead_Gyro_X_CAL,S_SeatHead_Gyro_Y_CAL,S_SeatHead_Gyro_Z_CAL,
ms,m/(s^2),m/(s^2),m/(s^2),deg/s,deg/s,deg/s,

The first and third lines should be ignored.

There will be an extra column of null data because of the trailing comma.

"""


def load_shimmer_file(path):
    """
    Parameters
    ==========
    path : string
        Path to a Shimmer csv file.

    Returns
    =======
    DataFrame

    """
    df = pd.read_csv(
        path,
        header=1,
        skiprows=[2],
        index_col=0,
        usecols=lambda x: x.startswith('S_'))  # skips dangling last column
    df.index = pd.to_datetime(df.index, unit='ms')
    return df


def load_session_files(session_label):
    """
    Parameters
    ==========
    session_label : string
        ``session001``, ``session002``, etc.

    Returns
    =======
    dictionary of DataFrame

    """
    with open(os.path.join(PATH_TO_REPO, 'data', 'sessions.yml')) as f:
        sessions = yaml.safe_load(f)
    file_names = sessions[session_label]['imu_files']
    data_frames = {}
    for label, filename in file_names.items():
        path_to_file = os.path.join(PATH_TO_DATA, 'Raw_data_csv', filename)
        data_frames[label] = load_shimmer_file(path_to_file)
    return data_frames


def merge_imu_data_frames(*data_frames):
    return None


if __name__ == "__main__":
    print(load_shimmer_file(PATH_TO_FILE))
