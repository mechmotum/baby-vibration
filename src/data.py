import os

import numpy as np
import pandas as pd
import yaml
from dtk.inertia import x_rot, y_rot, z_rot

with open('config.yml') as f:
    config_data = yaml.safe_load(f)

PATH_TO_DATA = config_data['data-directory']
PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


class Session():
    """Represents a continous period of data collection from multiple Shimmer
    IMUs, called a "session".

    Parameters
    ==========
    session_label : string
        ``sessionXXX`` where ``XXX`` is a three digit number ``000``, ``001``,
        etc. These labels are defined in ``data/session.yml``.

    """

    def __init__(self, session_label):

        self.session_label = session_label

        with open(os.path.join(PATH_TO_REPO, 'data', 'sessions.yml')) as f:
            sessions = yaml.safe_load(f)

        self.meta_data = sessions[session_label]

    def load_data(self):
        """Loads the IMU CSV files for this session into ``imu_data_frames``
        and the trial bound data into ``bounds_data_frame``."""

        path_to_bounds_file = os.path.join(PATH_TO_DATA, 'Interval_indexes',
                                           self.meta_data['trial_bounds_file'])

        self.imu_data_frames = load_session_files(self.session_label)
        self.bounds_data_frame = load_trial_bounds(
            path_to_bounds_file, self.imu_data_frames['rear_wheel'].index)

    def merge_imu_data(self):
        """Creates a single data frame, ``imu_data``, for all IMU data with NaN
        values at non-shared time stamps."""

        try:
            self.imu_data_frames
        except AttributeError:
            self.load_data()

        self.imu_data = merge_imu_data_frames(*self.imu_data_frames.values())

    def extract_trial(self, trial_name, trial_number=0):
        """Selects a trial from ``imu_data`` based on the manually defined
        bounds stored in ``bounds_data_frame``.

        Parameters
        ==========
        trial_name : string
            Examples are ``static``, ``Aula``, ``pave``, ``klinkers``, etc.
        trial_number : integer
            More than one trial with the same name may be present. Use this
            value to select the first, second, third, instance of that trial.

        Returns
        =======
        DataFrame
            Slice of ``imu_data``.

        """
        try:
            self.imu_data
        except AttributeError:
            self.merge_imu_data()

        count = 0
        for idx, row in self.bounds_data_frame.iterrows():
            if row['Surface'] == trial_name:
                if trial_number == count:
                    start_idx = row['start_time']
                    stop_idx = row['stop_time']
                    break
                count += 1

        return self.imu_data[start_idx:stop_idx]

    def rotate_imu_data(self, subtract_gravity=True):
        """Adds new columns to the ``imu_data`` data frame in which the
        accelerometer and rate gyro axes are rotated about the vehicle's
        lateral axis aligning one axis with the vertical (direction of gravity)
        and one with longitudinal.

        Notes
        =====
        """
        # TODO : Should I subtract the mean from the lateral axes? From the
        # rate gyro?
        df = self.extract_trial('static')
        mean_df = df.mean()
        rot_axis_labels = self.meta_data['imu_lateral_axis']
        for sensor, rot_axis_label in rot_axis_labels.items():
            template = 'S_{}_Accel_WR_{}_CAL'
            if rot_axis_label.endswith('x'):
                ver_lab, hor_lab = 'Y', 'Z'
                xyz = ('lat', 'ver', 'lon')
            elif rot_axis_label.endswith('y'):
                ver_lab, hor_lab = 'Z', 'X'
                xyz = ('lon', 'lat', 'ver')
            elif rot_axis_label.endswith('z'):
                ver_lab, hor_lab = 'X', 'Y'
                xyz = ('ver', 'lon', 'lat')
            ver_mean = mean_df[template.format(sensor, ver_lab)]
            hor_mean = mean_df[template.format(sensor, hor_lab)]
            rot_mat = compute_gravity_rotation_matrix(rot_axis_label,
                                                      ver_mean,
                                                      hor_mean)
            acc_cols = [col.format(sensor) for col in
                        ['S_{}_Accel_WR_X_CAL',
                         'S_{}_Accel_WR_Y_CAL',
                         'S_{}_Accel_WR_Z_CAL']]
            new_acc_cols = [col.format(sensor, d) for col, d in
                            zip(['{}acc_{}', '{}acc_{}', '{}acc_{}'], xyz)]
            self.imu_data[new_acc_cols] = (rot_mat @
                                           self.imu_data[acc_cols].values.T).T
            if subtract_gravity:
                # TODO : Change to taking the mean of the magnitude instead of
                # magnitude of the mean. Not sure if there would be a different
                # though.
                grav_acc = np.sqrt(np.sum(df[acc_cols].mean().values**2,
                                          axis=0))
                vert_col = '{}acc_ver'.format(sensor)
                self.imu_data[vert_col] += grav_acc
            gyr_cols = [col.format(sensor) for col in
                        ['S_{}_Gyro_X_CAL',
                         'S_{}_Gyro_Y_CAL',
                         'S_{}_Gyro_Z_CAL']]
            new_gyr_cols = [col.format(sensor, d) for col, d in
                            zip(['{}gyr_{}', '{}gyr_{}', '{}gyr_{}'], xyz)]
            self.imu_data[new_gyr_cols] = (rot_mat @
                                           self.imu_data[gyr_cols].values.T).T
        return self.imu_data


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


def load_trial_bounds(path, rw_timestamp):
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

    df['start_time'] = rw_timestamp[:len(df)]
    df['stop_time'] = rw_timestamp[:len(df)]

    for idx, row in df.iterrows():
        df.loc[idx, 'start_time'] = rw_timestamp.values[row['start_idx']]
        df.loc[idx, 'stop_time'] = rw_timestamp.values[row['stop_idx']]
    return df


def compute_gravity_rotation_matrix(lateral_axis, vertical_value,
                                    horizontal_value):
    """
    lateral_axis : string
        The sensor's axis that is approximately aligned with the vehicles
        lateral axis, either ``x``, ``y``, ``z``.
    vertical_value : float
        Standard gravity component in the body fixed axis that you desire to be
        aligned with gravity (vertical).
    horizontal_value : float
        Standard gravity component in the body fixed axis that you desire to be
        normal to gravity (horizontal).

    """
    rot_func = {'x': x_rot, 'y': y_rot, 'z': z_rot}
    theta = np.arctan(horizontal_value/vertical_value)
    if lateral_axis.startswith('-'):
        theta = -theta
        rot_mat = rot_func[lateral_axis[-1]](theta + np.pi/2)
    else:
        rot_mat = rot_func[lateral_axis[-1]](theta)
    print('Angle:', np.rad2deg(theta))
    return rot_mat


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
    s.rotate_imu_data()
    static = s.extract_trial('static')
    static.loc[:, static.columns.str.contains('Accel')].plot(subplots=True,
                                                             marker='.')
    static.loc[:, static.columns.str.contains('Gyro')].plot(subplots=True,
                                                            marker='.')
    static.loc[:, static.columns.str.contains('acc')].plot(subplots=True,
                                                           marker='.')
    #static.interpolate(method='time').plot(subplots=True)

    # convert time to float with s / pd.offsets.Second(1) or s =
    # pd.to_timedelta(pd.to_datetime(s)) or to_numeric

    plt.show()
