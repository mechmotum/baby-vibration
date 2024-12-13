import os

from dtk.inertia import x_rot, y_rot, z_rot
from dtk.process import freq_spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

with open('config.yml') as f:
    config_data = yaml.safe_load(f)

PATH_TO_SESSION_DATA = config_data['data-directory']
PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
PATH_TO_DATA_DIR = os.path.join(PATH_TO_REPO, 'data')
PATH_TO_FIG_DIR = os.path.join(PATH_TO_REPO, 'fig')


def magnitude(vector):
    """Calculates the magnitude of each vector in an array of vectors.

    Parameters
    ==========
    vector : array_like, shape(n, 3)
        n 3D vectors.

    Returns
    =======
    scalar : ndarray, shape(n,)
        Magnitude of n vectors.

    """
    # return np.sqrt(np.sum(vector**2, axis=1))
    return np.linalg.norm(vector, axis=1)


def datetime2seconds(index):
    """Converts a DateTimeIndex to seconds starting from zero.

    Parameters
    ==========
    index : DateTimeIndex, len(n)

    Returns
    =======
    time : ndarray, shape(n,)
        Time in seconds starting at 0 seconds.

    """

    time = (index.values -
            index.values.astype('datetime64[D]'))/np.timedelta64(1, 's')
    return time - time[0]


class Session():
    """Represents a continous period of data collection from multiple Shimmer
    IMUs, called a "session".

    Parameters
    ==========
    session_label : string
        ``sessionXXX`` where ``XXX`` is a three digit number ``000``, ``001``,
        etc. These labels are defined in ``data/session.yml``.

    """
    raw_gyr_tmpl = ['S_{}_Gyro_X_CAL', 'S_{}_Gyro_Y_CAL', 'S_{}_Gyro_Z_CAL']
    raw_acc_tmpl = ['S_{}_Accel_WR_X_CAL', 'S_{}_Accel_WR_Y_CAL',
                    'S_{}_Accel_WR_Z_CAL']

    def __init__(self, session_label):

        self.session_label = session_label

        with open(os.path.join(PATH_TO_DATA_DIR, 'sessions.yml')) as f:
            session_meta_data = yaml.safe_load(f)

        with open(os.path.join(PATH_TO_DATA_DIR, 'vehicles.yml')) as f:
            vehicle_meta_data = yaml.safe_load(f)

        self.meta_data = session_meta_data[session_label]
        self.meta_data.update(vehicle_meta_data[self.meta_data['vehicle']])

    def load_data(self):
        """Loads the IMU CSV files for this session into ``imu_data_frames``
        and the trial bound data into ``bounds_data_frame``."""

        self.imu_data_frames = load_session_files(self.session_label)

        self.trial_bounds = {}
        if self.meta_data['trial_bounds_file'] is None:
            self.bounds_data_frame = None
        else:
            path_to_bounds_file = os.path.join(
                PATH_TO_SESSION_DATA, 'Interval_indexes',
                self.meta_data['trial_bounds_file'])
            self.bounds_data_frame = load_trial_bounds(
                path_to_bounds_file, self.imu_data_frames['rear_wheel'].index)
            for trial_name in self.bounds_data_frame['Surface'].unique():
                selector = self.bounds_data_frame['Surface'].str.contains(
                    trial_name)
                counts = list(self.bounds_data_frame['count'][selector])
                self.trial_bounds[trial_name] = counts

    def merge_imu_data(self, minimize_memory=True):
        """Creates a single data frame, ``imu_data``, for all IMU data with NaN
        values at non-shared time stamps. Removes the individual data frames
        for each IMU in ``imu_data_frames`` unles ``minimize_memory=False``."""

        try:
            self.imu_data_frames
        except AttributeError:
            self.load_data()

        self.imu_data = merge_imu_data_frames(*self.imu_data_frames.values())

        if minimize_memory:
            # save memory by deleting this
            del self.imu_data_frames
            self.imu_data_frames = {}

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

        if trial_number not in self.trial_bounds[trial_name]:
            raise ValueError('Invalid trial number.')

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
            acc_cols = [col.format(sensor) for col in self.raw_acc_tmpl]
            new_acc_cols = [col.format(sensor, d) for col, d in
                            zip(['{}acc_{}', '{}acc_{}', '{}acc_{}'], xyz)]
            self.imu_data[new_acc_cols] = (rot_mat @
                                           self.imu_data[acc_cols].values.T).T
            if subtract_gravity:
                # TODO : Change to taking the mean of the magnitude instead of
                # magnitude of the mean. Not sure if there would be a different
                # though.
                mag_cols = ['S_{}_Accel_WR_{}_CAL'.format(sensor, ver_lab),
                            'S_{}_Accel_WR_{}_CAL'.format(sensor, hor_lab)]
                grav_acc = np.sqrt(np.sum(df[mag_cols].mean().values**2,
                                          axis=0))
                vert_col = '{}acc_ver'.format(sensor)
                self.imu_data[vert_col] += grav_acc
            gyr_cols = [col.format(sensor) for col in self.raw_gyr_tmpl]
            new_gyr_cols = [col.format(sensor, d) for col, d in
                            zip(['{}gyr_{}', '{}gyr_{}', '{}gyr_{}'], xyz)]
            self.imu_data[new_gyr_cols] = (rot_mat @
                                           self.imu_data[gyr_cols].values.T).T
        return self.imu_data

    def calculate_travel_speed(self):
        """Adds a column for the forward travel speed based on the angular rate
        gyro attached to the rotating wheel."""
        dia = self.meta_data['wheel_diameter']
        wheel_axis = self.meta_data['imu_lateral_axis']['RearWheel']
        if wheel_axis.startswith('-'):
            sign = 1.0
        else:
            sign = -1.0
        tmpl = 'S_RearWheel_Gyro_{}_CAL'
        ang_rate = self.imu_data[tmpl.format(wheel_axis[-1].upper())]
        self.imu_data['Speed'] = sign*np.deg2rad(ang_rate)*dia/2.0

    def calculate_vector_magnitudes(self):
        """Calculates the magnitudes of the acceleration and angular rate
        vectors for each IMU and adds a new column with the data."""
        for sensor in self.meta_data['imu_lateral_axis'].keys():
            # Use the gravity subtracted acceleration values.
            acc_cols = [col.format(sensor) for col in
                        ['{}acc_ver',
                         '{}acc_lon',
                         '{}acc_lat']]
            # Use raw data for the gyro instead of rotated.
            gyr_cols = [col.format(sensor) for col in self.raw_gyr_tmpl]

            # .values should return an array shape(n, 3)
            acc_mag = magnitude(self.imu_data[acc_cols].values)
            self.imu_data['{}acc_mag'.format(sensor)] = acc_mag

            gyr_mag = magnitude(self.imu_data[gyr_cols].values)
            self.imu_data['{}gyr_mag'.format(sensor)] = gyr_mag

    def differentiate_angular_rate(self):
        """Calculate the angular acceleration by filtering and numerical
        time differentiation.
        """
        # TODO
        pass

    def calculate_frequency_spectrum(self, signal, sample_rate, trial,
                                     trial_number=0):
        """Down samples and calculates the frequency spectrum."""
        data = self.extract_trial(trial, trial_number=trial_number)
        series = data[signal].dropna()
        time = datetime2seconds(series.index)
        signal = series.values
        deltat = 1.0/sample_rate
        new_time = np.arange(time[0], time[-1], deltat)
        new_signal = np.interp(new_time, time, signal)
        freq, amp = freq_spectrum(new_signal, sample_rate)
        return freq, amp

    def plot_speed_with_trial_bounds(self):
        """Createas a plot of forward spee versus time for the whole session
        with shaded labeled areas for each trial."""
        fig, ax = plt.subplots(figsize=(16, 2), layout='constrained')
        ax = self.imu_data['Speed'].plot(ax=ax, linestyle='', marker='.')
        for idx, row in self.bounds_data_frame.iterrows():
            ax.axvspan(row['start_time'], row['stop_time'],
                       alpha=0.5, color='gray')
            ax.text(row['start_time'], 0.0, row['Surface'],
                    rotation='vertical')
        ax.set_ylabel('Speed [m/s]')
        ax.set_title(self.meta_data['imu_files']['rear_wheel'])
        return ax

    def plot_raw_time_series(self, trial=None, trial_number=0, acc=True,
                             gyr=True):
        """Returns a plot of the raw acelerometer and gyroscope time series.

        Parameters
        ==========
        trial : string
        trial_number : integer
        acc : boolean
            If true, plot includes accelerometer data.
        gyr : boolean
            If true, plot includes gyroscope data.

        """
        if trial is not None:
            data = self.extract_trial(trial, trial_number=trial_number)
        else:
            data = self.imu_data

        if acc and gyr:
            # all raw data acc & gyr data begins with S_
            cols = data.columns.str.startswith('S_')
        elif acc:
            cols = data.columns.str.contains('Accel')
        elif gyr:
            cols = data.columns.str.contains('Gyro')
        else:
            raise ValueError('One of acc or gyr must be True.')

        subset = data.loc[:, cols]

        return subset.plot(subplots=True, marker='.')


def plot_frequency_spectrum(freq, amp, rms, sample_rate):
    """Returns plot of the amplitude versus frequency for the freqeuncy range
    of the sample rate / 2."""
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(freq, amp)
    ax.axhline(rms, color='black')
    ax.set_xlim((0.0, sample_rate/2.0))
    ax.set_ylim((0.0, 2.0))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [m/s/s]')
    ax.grid()
    return ax


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

    In later files the first three lines can look like this::

        S_BotTrike_Timestamp_Unix_CAL,S_BotTrike_Accel_WR_X_CAL,S_BotTrike_Accel_WR_Y_CAL,S_BotTrike_Accel_WR_Z_CAL,S_BotTrike_Gyro_X_CAL,S_BotTrike_Gyro_Y_CAL,S_BotTrike_Gyro_Z_CAL
        ms,m/(s^2),m/(s^2),m/(s^2),deg/s,deg/s,deg/s
        S_BotTrike_Timestamp_Unix_CAL,S_BotTrike_Accel_WR_X_CAL,S_BotTrike_Accel_WR_Y_CAL,S_BotTrike_Accel_WR_Z_CAL,S_BotTrike_Gyro_X_CAL,S_BotTrike_Gyro_Y_CAL,S_BotTrike_Gyro_Z_CAL

    """
    with open(path, 'r') as f:
        first_line = f.readline()
    if first_line.startswith('"sep='):
        df = pd.read_csv(
            path,
            header=1,
            skiprows=[2],
            index_col=0,
            usecols=lambda x: x.startswith('S_'))  # skips dangling last column
    else:
        df = pd.read_csv(
            path,
            header=0,
            skiprows=[1, 2],
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
        path_to_file = os.path.join(PATH_TO_SESSION_DATA, 'Raw_data_csv',
                                    filename)
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

    df['count'] = [0]*len(df)

    counts = {}
    for idx, row in df.iterrows():
        if row['Surface'] in counts:
            counts[row['Surface']] += 1
        else:
            counts[row['Surface']] = 0
        df.loc[idx, 'start_time'] = rw_timestamp.values[row['start_idx']]
        df.loc[idx, 'stop_time'] = rw_timestamp.values[row['stop_idx']]
        df.loc[idx, 'count'] = counts[row['Surface']]
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

    session_label = 'session001'

    s = Session(session_label)
    s.merge_imu_data()
    s.rotate_imu_data()
    s.calculate_travel_speed()
    s.calculate_vector_magnitudes()
    freq, amp = s.calculate_frequency_spectrum('SeatBotacc_mag', 200,
                                               trial='Aula')
    rms = 2.0*np.sqrt(np.mean(amp**2))
    # static = s.extract_trial('Aula')
    # plot_frequency_spectrum(freq, amp, rms, 200)
    # s.plot_raw_time_series(trial='Aula', gyr=False)
    # s.plot_raw_time_series(trial='Aula', acc=False)
    # static.loc[:, static.columns.str.contains('acc')].plot(
    #     subplots=True, marker='.')
    # static.interpolate(method='time').plot(subplots=True)

    # convert time to float with s / pd.offsets.Second(1) or s =
    # pd.to_timedelta(pd.to_datetime(s)) or to_numeric

    # s.plot_raw_time_series(trial='stoeptegels', gyr=False)
    # s.plot_raw_time_series(trial='stoeptegels', acc=False)
    # s.plot_raw_time_series(gyr=False)

    s.plot_speed_with_trial_bounds()

    plt.show()
