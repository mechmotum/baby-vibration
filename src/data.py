import os
import gc

from dtk.inertia import x_rot, y_rot, z_rot
from dtk.process import freq_spectrum, butterworth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from paths import PATH_TO_SESSION_DATA, PATH_TO_DATA_DIR
from functions import (load_session_files, load_trial_bounds,
                       load_trial_bounds2, merge_imu_data_frames,
                       compute_gravity_rotation_matrix, magnitude,
                       datetime2seconds)
from plots import plot_frequency_spectrum


class Session():
    """Represents a continous period of data collection from multiple Shimmer
    IMUs, called a "session".

    Parameters
    ==========
    session_label : string
        ``sessionXXX`` where ``XXX`` is a three digit number ``000``, ``001``,
        etc. These labels are defined in ``data/session.yml``.

    """
    sensor_labels = ['BotTrike', 'FrontWheel', 'RearWheel', 'SeatBot',
                     'SeatHead']
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

        if trial_name not in self.trial_bounds:
            msg = 'Trial type {} not present in this session'
            raise ValueError(msg.format(trial_name))

        if trial_number not in self.trial_bounds[trial_name]:
            msg = 'Trial number {} not present in available repititions: {}'
            raise ValueError(msg.format(trial_number,
                                        self.trial_bounds[trial_name]))

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
                                     trial_number=0, iso_weighted=False,
                                     smooth=False):
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

        if smooth:
            f = 1/(freq[1] - freq[0])
            amp = butterworth(amp, f/50, f)

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

    def plot_time_sync(self):
        # The longitudinal acceleration of each sensor except the rotating rear
        # wheel sensor should output the same thing during the time
        # synchronization motion.
        data = self.extract_trial('synchro')
        cols = ['{}acc_lon'.format(sensor)
                for sensor in self.sensor_labels if sensor != 'RearWheel']
        lon_acc = data[cols].interpolate(method='time')
        return lon_acc.plot()

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


if __name__ == "__main__":

    session_label = 'session001'
    trial_label = 'static'
    sample_rate = 400

    s = Session(session_label)
    s.load_data()
    s.merge_imu_data()
    s.rotate_imu_data(subtract_gravity=False)
    s.calculate_travel_speed()
    s.calculate_vector_magnitudes()

    s.plot_accelerometer_rotation()

    s.rotate_imu_data()
    if 'synchro' in s.trial_bounds:
        s.plot_time_sync()
    s.plot_speed_with_trial_bounds()
    s.plot_raw_time_series(trial=trial_label, gyr=False)
    s.plot_raw_time_series(trial=trial_label, acc=False)
    s.plot_iso_weights()

    freq, amp, tim, sig = s.calculate_frequency_spectrum(
        'SeatBotacc_ver', sample_rate, trial_label)
    rms_time = np.sqrt(np.mean(sig**2))
    print('Unweighted RMS from time domain: ', rms_time)
    ax = plot_frequency_spectrum(freq, amp)
    freq, amp, _, _ = s.calculate_frequency_spectrum(
        'SeatBotacc_ver', sample_rate, trial_label, smooth=True)
    plot_frequency_spectrum(freq, amp, ax=ax, plot_kw={'linewidth': 4})

    plt.show()
