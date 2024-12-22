import os
import gc

from dtk.inertia import x_rot, y_rot, z_rot
from dtk.process import freq_spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from paths import PATH_TO_SESSION_DATA, PATH_TO_REPO, PATH_TO_DATA_DIR


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

        filter_data_01 = os.path.join(PATH_TO_DATA_DIR,
                                      'iso-2631-filter-01.csv')
        self.iso_filter_df_01 = pd.read_csv(filter_data_01,
                                            index_col='frequency_hz')
        filter_data_02 = os.path.join(PATH_TO_DATA_DIR,
                                      'iso-2631-filter-02.csv')
        self.iso_filter_df_02 = pd.read_csv(filter_data_02,
                                            index_col='frequency_hz')

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
            if self.meta_data['trial_bounds_file'].startswith('events'):
                self.bounds_data_frame = load_trial_bounds2(
                    path_to_bounds_file)
            else:
                self.bounds_data_frame = load_trial_bounds(
                    path_to_bounds_file,
                    self.imu_data_frames['rear_wheel'].index)

            for trial_name in self.bounds_data_frame['surface'].unique():
                selector = self.bounds_data_frame['surface'].str.contains(
                    trial_name)
                counts = list(self.bounds_data_frame['count'][selector])
                self.trial_bounds[trial_name] = counts

    def memory_usage(self):
        msg = 'imu_data data frame : {:0.2f} bytes'.format(
            self.imu_data.memory_usage().sum())
        print(msg)

    def merge_imu_data(self, minimize_memory=True):
        """Creates a single data frame, ``imu_data``, for all IMU data with NaN
        values at non-shared time stamps. Removes the individual data frames
        for each IMU in ``imu_data_frames`` unles ``minimize_memory=False``."""

        try:
            self.imu_data_frames
        except AttributeError:
            self.load_data()

        # TODO : Could store this as a sparse data type, but there are
        # failures, such as .interpolate() not being available.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html
        # .astype(pd.SparseDtype("float", np.nan))

        self.imu_data = merge_imu_data_frames(*self.imu_data_frames.values())

        if minimize_memory:
            # save memory by deleting this
            del self.imu_data_frames
            gc.collect()
            self.imu_data_frames = {}

    def extract_trial(self, trial_name, trial_number=0):
        """Selects a trial from ``imu_data`` based on the manually defined
        bounds stored in ``bounds_data_frame``.

        Parameters
        ==========
        trial_name : string
            Examples are ``static``, ``aula``, ``pave``, ``klinkers``, etc.
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

        surf_crit = self.bounds_data_frame['surface'] == trial_name
        count_crit = self.bounds_data_frame['count'] == trial_number
        row = self.bounds_data_frame[surf_crit & count_crit]

        start_idx = row['start_time'].values[0]
        stop_idx = row['end_time'].values[0]

        return self.imu_data[start_idx:stop_idx]

    def rotate_imu_data(self, subtract_gravity=True):
        """Adds new columns to the ``imu_data`` data frame in which the
        accelerometer and rate gyro axes are rotated about the vehicle's
        lateral axis aligning one axis with the vertical (direction of gravity)
        and one with longitudinal.

        Notes
        =====
        """
        raw_acc_tmpl = 'S_{}_Accel_WR_{}_CAL'

        # TODO : Should I subtract the mean from the lateral axes? From the
        # rate gyro?
        df = self.extract_trial('static')
        mean_df = df.mean()

        rot_axis_labels = self.meta_data['imu_lateral_axis']
        for sensor, rot_axis_label in rot_axis_labels.items():
            if rot_axis_label.endswith('x'):
                ver_lab, hor_lab = 'Y', 'Z'
                xyz = ('lat', 'ver', 'lon')
            elif rot_axis_label.endswith('y'):
                ver_lab, hor_lab = 'Z', 'X'
                xyz = ('lon', 'lat', 'ver')
            elif rot_axis_label.endswith('z'):
                ver_lab, hor_lab = 'X', 'Y'
                xyz = ('ver', 'lon', 'lat')
            hor_mean = mean_df[raw_acc_tmpl.format(sensor, hor_lab)]
            ver_mean = mean_df[raw_acc_tmpl.format(sensor, ver_lab)]
            #print('Sensor', sensor)
            #print(hor_mean, ver_mean)
            rot_mat = compute_gravity_rotation_matrix(rot_axis_label,
                                                      ver_mean,
                                                      hor_mean)
            acc_cols = [col.format(sensor) for col in self.raw_acc_tmpl]
            new_acc_cols = [col.format(sensor, d) for col, d in
                            zip(['{}acc_{}', '{}acc_{}', '{}acc_{}'], xyz)]
            self.imu_data[new_acc_cols] = (
                rot_mat @ self.imu_data[acc_cols].values.T).T

            gyr_cols = [col.format(sensor) for col in self.raw_gyr_tmpl]
            new_gyr_cols = [col.format(sensor, d) for col, d in
                            zip(['{}gyr_{}', '{}gyr_{}', '{}gyr_{}'], xyz)]
            self.imu_data[new_gyr_cols] = (
                rot_mat @ self.imu_data[gyr_cols].values.T).T
            # NOTE : This is a bit shameful that I can't figure out the correct
            # rotation to apply and just apply a brute force check and 180
            # rotation.
            #print('Mean of vert',
                  #self.imu_data['{}acc_ver'.format(sensor)].mean())
            if self.imu_data['{}acc_ver'.format(sensor)].mean() < 0.0:
                rot_func = {'x': x_rot, 'y': y_rot, 'z': z_rot}
                rot_mat = rot_func[rot_axis_label[-1]](np.pi)
                self.imu_data[new_acc_cols] = (
                    rot_mat @ self.imu_data[new_acc_cols].values.T).T
                self.imu_data[new_gyr_cols] = (
                    rot_mat @ self.imu_data[new_gyr_cols].values.T).T

            if subtract_gravity:
                # TODO : Change to taking the mean of the magnitude instead of
                # magnitude of the mean. Not sure if there would be a different
                # though.
                mag_cols = [raw_acc_tmpl.format(sensor, ver_lab),
                            raw_acc_tmpl.format(sensor, hor_lab)]
                grav_acc = np.sqrt(np.sum(df[mag_cols].mean().values**2,
                                          axis=0))
                vert_col = '{}acc_ver'.format(sensor)
                self.imu_data[vert_col] -= grav_acc

        del df
        gc.collect()

        return self.imu_data

    def calculate_travel_speed(self, smooth=False):
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
        if smooth:
            smoothed = self.imu_data['Speed'].rolling('100ms').mean()
            self.imu_data['Speed'] = smoothed
        self.imu_data['Speed_kph'] = 3.6*self.imu_data['Speed']

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

    def calculate_frequency_spectrum(self, sig_name, sample_rate, trial,
                                     trial_number=0, iso_weighted=False):
        """Down samples and calculates the frequency spectrum."""
        data = self.extract_trial(trial, trial_number=trial_number)
        series = data[sig_name].dropna()
        time = datetime2seconds(series.index)
        signal = series.values
        deltat = 1.0/sample_rate
        new_time = np.arange(time[0], time[-1], deltat)
        new_signal = np.interp(new_time, time, signal)
        new_signal -= np.mean(new_signal)

        freq, amp = freq_spectrum(new_signal, sample_rate)

        if iso_weighted:
            table_freq = self.iso_filter_df_01.index.values
            if 'acc' in sig_name:
                if 'acc_ver' in sig_name:
                    col = 'vertical_acceleration_z'
                else:
                    col = 'translational_acceleration_xy'
            elif 'gyr' in signal:
                col = 'rotation_speed_xyz'
            else:
                raise ValueError('no weights!')
            table_weights = self.iso_filter_df_01[col].values/1000.0
            weights = np.interp(freq, table_freq, table_weights)
            amp = weights*amp

        del data
        gc.collect()

        return freq, amp, time, new_signal

    def plot_speed_with_trial_bounds(self):
        """Createas a plot of forward speed versus time for the whole session
        with shaded labeled areas for each trial."""
        fig, ax = plt.subplots(figsize=(16, 3), layout='constrained')
        ax = self.imu_data['Speed_kph'].plot(ax=ax, linestyle='', marker='.')
        for idx, row in self.bounds_data_frame.iterrows():
            start, end = row['start_time'], row['end_time']

            chunk = self.imu_data.loc[start:end, 'Speed_kph']
            mean, std = chunk.mean(), chunk.std()
            ax.plot([start, end], [mean, mean], color='gold')
            ax.plot([start, end], [mean + std, mean + std], color='khaki')
            ax.plot([start, end], [mean - std, mean - std], color='khaki')
            del chunk

            ax.axvspan(start, row['end_time'], alpha=0.5, color='gray')
            ax.text(start, 0.0, row['surface'], rotation='vertical')

        ax.set_ylabel('Speed [km/h]')
        ax.set_title(self.meta_data['imu_files']['rear_wheel'])
        ax.grid()
        return ax

    def plot_accelerometer_rotation(self):
        data = self.extract_trial('static')
        fig, axes = plt.subplots(15, 2, layout='constrained', sharex=True,
                                 figsize=(10, 15))
        raw_acc_labels = []
        rot_acc_labels = []
        selector = {'x': 0, 'y': 1, 'z': 2}
        imus = self.meta_data['imu_lateral_axis'].items()
        for snum, (sensor, axis) in enumerate(imus):
            raw_acc_labels += [tmpl.format(sensor) for tmpl in
                               self.raw_acc_tmpl]

            idx = 3*snum + selector[axis[-1]]
            axes[idx, 0].set_facecolor('gray')
            axes[idx, 0].set_title(axis)
            rot_acc_labels += [tmpl.format(sensor) for tmpl in
                               ['{}acc_ver', '{}acc_lat', '{}acc_lon']]
            if data['{}acc_ver'.format(sensor)].mean() < 0.0:
                axes[3*snum, 1].set_facecolor('cornsilk')

        data = data.interpolate(method='time')
        data[raw_acc_labels].plot(subplots=True, ax=axes[:, 0])
        data[rot_acc_labels].plot(subplots=True, ax=axes[:, 1])
        for ax in axes.flatten():
            ax.set_ylim((-12, 12))

        del data
        gc.collect()
        return axes

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

    def plot_iso_weights(self):
        ax1 = self.iso_filter_df_01.plot(subplots=True)
        ax1[0].set_title('iso_filter_01')
        ax2 = self.iso_filter_df_02.plot(subplots=True)
        ax2[0].set_title('iso_filter_02')
        return ax1, ax2


def plot_frequency_spectrum(freq, amp, rms, sample_rate, ax=None):
    """Returns plot of the amplitude versus frequency for the freqeuncy range
    of the sample rate / 2."""
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    ax.plot(freq, amp)
    ax.axhline(rms, color=ax.get_lines()[0].get_color())
    ax.set_ylim((0.0, 1.0))
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
    df[['start_idx', 'end_idx']] = df['RearWheel'].str.split(',', expand=True)
    df['start_idx'] = df['start_idx'].str.replace('[', '').astype(int)
    df['end_idx'] = df['end_idx'].str.replace(']', '').astype(int)

    df['start_time'] = rw_timestamp[:len(df)]
    df['end_time'] = rw_timestamp[:len(df)]

    df['count'] = [0]*len(df)

    counts = {}
    for idx, row in df.iterrows():
        if row['Surface'] in counts:
            counts[row['Surface']] += 1
        else:
            counts[row['Surface']] = 0
        df.loc[idx, 'start_time'] = rw_timestamp.values[row['start_idx']]
        df.loc[idx, 'end_time'] = rw_timestamp.values[row['end_idx']]
        df.loc[idx, 'count'] = counts[row['Surface']]

    df.rename(columns={'Surface': 'surface'}, inplace=True)

    df['surface'] = df['surface'].str.lower()
    return df


def load_trial_bounds2(path):
    df = pd.read_csv(path, index_col='segment_number')
    time_cols = ['start_time', 'end_time']
    df[time_cols] = df[time_cols].apply(
        lambda x: pd.to_datetime(x, unit='ms'))

    # NOTE : _12kph is appended to many surface names and needs to be removed
    df['surface'] = df['surface'].str.split('_', expand=True)[0]
    # NOTE : the shocks are numbered shock1, shock2 so remove
    df.loc[df['surface'].str.contains('shock'), 'surface'] = 'shock'

    df['count'] = [0]*len(df)
    counts = {}
    for idx, row in df.iterrows():
        if row['surface'] in counts:
            counts[row['surface']] += 1
        else:
            counts[row['surface']] = 0
        df.loc[idx, 'count'] = counts[row['surface']]

    df['surface'] = df['surface'].str.lower()
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
    rot_mat = rot_func[lateral_axis[-1]](theta)
    #print('Angle:', np.rad2deg(theta))
    return rot_mat


if __name__ == "__main__":

    session_label = 'session001'
    trial_label = 'aula'
    sample_rate = 400

    s = Session(session_label)
    s.load_data()
    s.merge_imu_data()
    s.rotate_imu_data(subtract_gravity=False)
    s.calculate_travel_speed()
    s.calculate_vector_magnitudes()

    s.plot_accelerometer_rotation()

    s.rotate_imu_data()
    s.plot_speed_with_trial_bounds()
    s.plot_raw_time_series(trial=trial_label, gyr=False)
    s.plot_raw_time_series(trial=trial_label, acc=False)
    s.plot_iso_weights()

    freq, amp, tim, sig = s.calculate_frequency_spectrum(
        'SeatBotacc_ver', sample_rate, trial_label)
    # TODO : I think this rms must be wrong for our scaling approach.
    rms_spec = np.sqrt(2.0*np.mean(amp**2))
    print('Unweighted RMS from frequency spectrum: ', rms_spec)
    rms_time = np.sqrt(np.mean(sig**2))
    print('Unweighted RMS from time domain: ', rms_time)

    plot_frequency_spectrum(freq, amp, rms_spec, sample_rate)

    freq, amp, _, _ = s.calculate_frequency_spectrum(
        'SeatBotacc_ver', sample_rate, trial_label, iso_weighted=True)
    rms_spec = np.sqrt(2.0*np.mean(amp**2))
    print('Weighted RMS from frequency spectrum: ', rms_spec)
    rms_time = np.sqrt(np.mean(sig**2))
    # TODO : Make this weighted by doing an inverse FFT.
    print('Unweighted RMS from time domain: ', rms_time)
    plot_frequency_spectrum(freq, amp, rms_spec, sample_rate)

    plt.show()
