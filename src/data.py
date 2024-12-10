import os

import numpy as np
import pandas as pd
import yaml

with open('config.yml') as f:
    config_data = yaml.safe_load(f)

PATH_TO_DATA = config_data['data-directory']
PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


class Session():
    def __init__(self, session_label):

        self.session_label = session_label

        with open(os.path.join(PATH_TO_REPO, 'data', 'sessions.yml')) as f:
            sessions = yaml.safe_load(f)

        self.meta_data = sessions[session_label]

    def load_data(self):

        path_to_bounds_file = os.path.join(PATH_TO_DATA, 'Interval_indexes',
                                           self.meta_data['trial_bounds_file'])

        self.imu_data_frames = load_session_files(self.session_label)
        self.bounds_data_frame = load_trial_bounds(
            path_to_bounds_file, self.imu_data_frames['rear_wheel'].index)

    def merge_data(self):

        try:
            self.imu_data_frames
        except AttributeError:
            self.load_data()

        self.raw_data = merge_imu_data_frames(*self.imu_data_frames.values())

    def extract_trial(self, trial_name, trial_number=0):

        try:
            self.raw_data
        except AttributeError:
            self.merge_data()

        count = 0
        for idx, row in self.bounds_data_frame.iterrows():
            if row['Surface'] == trial_name:
                if trial_number == count:
                    start_idx = row['start_time']
                    stop_idx = row['stop_time']
                    break
                count += 1

        return self.raw_data[start_idx:stop_idx]


def load_shimmer_file(path):
    """Loads a single Shimmer aquisition csv file into Pandas data frame.

    Parameters
    ==========
    path : string
        Path to a Shimmer csv file.

    Returns
    =======
    DataFrame

    Notes
    =====
    Assumes first column is the time stamp, first row is garbage, second row is
    the measure name, and third row is the units.

    First three lines look like this::

       "sep=,"
       S_SeatHead_Timestamp_Unix_CAL,S_SeatHead_Accel_WR_X_CAL,S_SeatHead_Accel_WR_Y_CAL,S_SeatHead_Accel_WR_Z_CAL,S_SeatHead_Gyro_X_CAL,S_SeatHead_Gyro_Y_CAL,S_SeatHead_Gyro_Z_CAL,
       ms,m/(s^2),m/(s^2),m/(s^2),deg/s,deg/s,deg/s,

    The first and third lines should be ignored. There will be an extra column
    of null data because of the trailing comma.

    """
    df = pd.read_csv(
        path,
        header=1,
        skiprows=[2],
        index_col=0,
        usecols=lambda x: x.startswith('S_'))  # skips dangling last column
    df.index = pd.to_datetime(df.index, unit='ms')
    df.index.name = 'Timestamp'
    return df


def load_session_files(session_label):
    """Loads all Shimmer IMU acquisition files associated with a single session
    based on the session label defined in ``data/sessions.yml``.

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
    """Combines all Shimmer IMU aquisition data frames into a single data
    frame. This assumes that the time stamps are syncronized in real time.

    """
    merged = data_frames[0]
    for df_next in data_frames[1:]:
        merged = pd.merge(merged, df_next, left_index=True, right_index=True,
                          how='outer')
    return merged


def load_trial_bounds(path, rear_wheel_timestamp):
    """Returns a data frame that has a row for each trial in a session along
    with its start and stop indices.

Surface,BotTrike,SeatBot,SeatHead,RearWheel,FrontWheel
static,"[148116, 151684]",,,,
static,,"[156665, 160216]",,,
static,,,"[161435, 165003]",,
static,,,,"[119249, 122815]",
static,,,,,"[124468, 128020]"
Aula,"[65784, 83246]",,,,
    """
    df = pd.read_csv(path, header=0)
    # only use the rear wheel indices
    df = df[['Surface', 'RearWheel']].dropna()
    # Index,Surface,RearWheel
    # 0,static,"[119249, 122815]"
    df.index = range(len(df))
    df[['start_idx', 'stop_idx']] = df['RearWheel'].str.split(',', expand=True)
    df['start_idx'] = df['start_idx'].str.replace('[', '').astype(int)
    df['stop_idx'] = df['stop_idx'].str.replace(']', '').astype(int)

    df['start_time'] = rear_wheel_timestamp[:len(df)]
    df['stop_time'] = rear_wheel_timestamp[:len(df)]

    for idx, row in df.iterrows():
        df.loc[idx, 'start_time'] = rear_wheel_timestamp.values[row['start_idx']]
        df.loc[idx, 'stop_time'] = rear_wheel_timestamp.values[row['stop_idx']]
    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    session_label = 'session001'

    #dfs = load_session_files(session_label)

    #dfs['front_wheel'].plot(subplots=True)
    #dfs['rear_wheel'].plot(subplots=True)
    #dfs['seat_bottom'].plot(subplots=True)
    #dfs['trike_bottom'].plot(subplots=True)
    #dfs['seat_head'].plot(subplots=True)

    #merged = merge_imu_data_frames(*dfs.values())

    # NOTE : The missing values (NANs) cause this plot to show gaps. I think it
    # is simply a consequence of plotting the NANs, but it is a bit confusing.
    #merged.plot(subplots=True, linestyle='-')
    #merged.plot(subplots=True, marker='.')

    # NOTE : This fills all NANs with linear interpolation based on the time
    # index.
    # TODO : check if there is a difference in method='time' or 'index'
    #interpolated = merged.interpolate(method='time')
    #interpolated = merged.interpolate(method='index')
    #interpolated.plot(subplots=True)

    #with open(os.path.join(PATH_TO_REPO, 'data', 'sessions.yml')) as f:
        #sessions = yaml.safe_load(f)
    #filename = sessions[session_label]['trial_bounds_file']
    #path_to_file = os.path.join(PATH_TO_DATA, 'Interval_indexes', filename)
    #bounds_df = load_trial_bounds(path_to_file, dfs['rear_wheel'].index)
    #static = merged[bounds_df.loc[0, 'start_time']:bounds_df.loc[0, 'stop_time']]
    #static.plot(subplots=True, marker='.')
    s = Session(session_label)
    static = s.extract_trial('static')
    static.loc[:, static.columns.str.contains('Accel')].plot(subplots=True, marker='.')
    #static.interpolate(method='time').plot(subplots=True)

    plt.show()
