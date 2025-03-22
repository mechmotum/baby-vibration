# builtin
import datetime
import functools
import gc
import os
import pprint

# dependencies
from dtk.inertia import x_rot, y_rot, z_rot
from dtk.process import freq_spectrum, butterworth
from scipy.integrate import cumulative_trapezoid, trapezoid
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# local
from functions import (load_session_files, load_trial_bounds,
                       load_trial_bounds2, merge_imu_data_frames,
                       compute_gravity_rotation_matrix, magnitude,
                       datetime2seconds, header, to_dense)
from paths import PATH_TO_DATA_DIR
from plots import plot_frequency_spectrum, plot_iso_weights

# NOTE : This seems to reduce memory consumption.
pd.options.mode.copy_on_write = True

filter_data_01 = os.path.join(PATH_TO_DATA_DIR, 'iso-2631-filter-01.csv')
iso_filter_df_01 = pd.read_csv(filter_data_01, index_col='frequency_hz')
filter_data_02 = os.path.join(PATH_TO_DATA_DIR, 'iso-2631-filter-02.csv')
iso_filter_df_02 = pd.read_csv(filter_data_02, index_col='frequency_hz')


class Trial():
    """Represents a single trial. A trial is time continuous set of IMU data
    taken from a specific scenario that occured in a session. The data should
    contain all columns that must be computed when all information from the
    session is available, i.e. run ``Session``'s processing and then extract
    trials with ``Session.extract_trial()``.

    Parameters
    ==========
    meta_data : dictionary
    imu_data : DataFrame
        The index is a ``DatetimeIndex`` and the columns correspond to data
        collected at those times.

    """
    def __init__(self, meta_data, imu_data):
        self.meta_data = meta_data
        self.imu_data = imu_data

    def __str__(self):
        return pprint.pformat(self.meta_data)

    @functools.cache
    def down_sample(self, sig_name, sample_rate, cutoff=None):
        """Returns the data in the column interpolated at a constant sample
        rate.

        Parameters
        ==========
        sig_name : string
            Column name to down sample.
        sample_rate : integer
            Sample rate in Hertz.
        cutoff : None or float
            Low pass cutoff frequency for the optionally applied filter.

        Returns
        =======
        new_time : ndarray, shape(n,)
            Time array in seconds starting at 0.0 with constant spacing at the
            sample rate.
        new_signal : ndarray, shape(n,)
            Down sampled (and possibly filtered) signal corresponding to the
            time values in ``new_time``.

        """
        series = self.imu_data[sig_name].dropna()
        time = datetime2seconds(series.index)
        signal = series.values
        deltat = 1.0/sample_rate
        new_time = np.arange(time[0], time[-1], deltat)
        new_signal = np.interp(new_time, time, signal)
        # TODO : These should be applied to the whole data frame in Session.
        # Caps for sensor limits.
        if 'acc' in sig_name.lower():
            new_signal[new_signal > 16.0*9.81] = 16.0*9.81
            new_signal[new_signal < -16.0*9.81] = -16.0*9.81
        elif 'gyr' in sig_name.lower():
            new_signal[new_signal > 2000.0] = 2000.0
            new_signal[new_signal < -2000.0] = -2000.0

        if cutoff is not None:
            new_signal = butterworth(new_signal, cutoff, sample_rate)
        return new_time, new_signal

    @functools.cache
    def calculate_frequency_spectrum(self, sig_name, sample_rate, cutoff=None,
                                     iso_weighted=False, smooth=False):
        """Down samples to a constant sample rate and and returns the frequency
        spectrum using an FFT.

        Parameters
        ==========
        sig_name : string
            Column name to down sample.
        sample_rate : integer
            Sample rate in Hertz.
        cutoff : None or float
            If a float is supplied the signal will be downsampled and low pass
            filtered at the provided cutoff frequency in Hz before the FFT is
            applied.
        iso_weighted : boolean
            If true, spectrum will be weighted using the filters in ISO 2631-1.
        smooth : boolean
            If true, the spectrum will be filtered using a 2nd order zero-lag
            Butterworth filter.

        Returns
        =======
        freq : ndarray, shape(sample_rate,)
            Frequency in Hertz.
        amp : ndarray, shape(sample_rate,)
            Fourier amplitude at each frequency. Units are the same as the
            processed signal.
        new_time : ndarray, shape(n,)
            Time array in seconds starting at 0.0 with constant spacing at the
            sample rate.
        new_signal : ndarray, shape(n,)
            Down sampled and mean subtracted signal corresponding to the time
            values in ``new_time``.

        """

        new_time, new_signal = self.down_sample(sig_name, sample_rate,
                                                cutoff=cutoff)

        new_signal -= np.mean(new_signal)

        freq, amp = freq_spectrum(new_signal, sample_rate)

        if smooth:
            f = 1/(freq[1] - freq[0])
            amp = butterworth(amp, f/80, f)

        if iso_weighted:
            table_freq = iso_filter_df_01.index.values
            if 'acc' in sig_name:
                if 'acc_ver' in sig_name:
                    col = 'vertical_acceleration_z'
                else:
                    col = 'translational_acceleration_xy'
            elif 'gyr' in sig_name:
                col = 'rotation_speed_xyz'
            else:
                raise ValueError('no weights!')
            table_weights = iso_filter_df_01[col].values/1000.0
            weights = np.interp(freq, table_freq, table_weights)
            amp = weights*amp

        return freq, amp, new_time, new_signal

    def plot_frequency_spectrum(self, sig_name, sample_rate, cutoff=None,
                                iso_weighted=False, smooth=False, ax=None,
                                show_features=False):
        """Down samples to a constant sample rate and and returns a plot of the
        frequency spectrum using an FFT.

        Parameters
        ==========
        sig_name : string
            Column name to down sample.
        sample_rate : integer
            Sample rate in Hertz.
        cutoff : None or float
            If a float is supplied the signal will be downsampled and low pass
            filtered at the provided cutoff frequency in Hz before the FFT is
            applied.
        iso_weighted : boolean
            If true, amplitude spectrum will be weighted using the filters in
            ISO 2631-1 before FFT smoothing.
        smooth : boolean
            If true, the amplitude spectrum will be filtered using a 2nd order
            zero-lag Butterworth filter.
        ax : Axis
            Matplotlib axis to plot on.
        show_features : boolean or string
            If true, shows peak frequency and bandwidth threshold frequency. If
            ``'bandwidth from smooth'``, bandwidth is calculated before ISO
            weighting.

        """

        # raw data
        freq, amp, _, _ = self.calculate_frequency_spectrum(
            sig_name, sample_rate, cutoff=cutoff, iso_weighted=False,
            smooth=False)
        ax = plot_frequency_spectrum(freq, amp, ax=ax,
                                     plot_kw={'color': 'gray',
                                              'linewidth': 0.5,
                                              'alpha': 0.8})
        legend = ['Raw FFT']

        # features are from fully processed, but plotted first
        if show_features:
            max_amp, peak_freq, thresh_freq = self.calc_spectrum_features(
                sig_name, sample_rate, cutoff=cutoff,
                iso_weighted=iso_weighted, smooth=smooth)
            if smooth:
                color = 'C0'
                thresh_color = 'C0'
            else:
                color = 'black'
                thresh_color = 'black'
            if show_features == 'bandwidth from smooth':
                _, _, thresh_freq = self.calc_spectrum_features(
                    sig_name, sample_rate, cutoff=cutoff,
                    iso_weighted=False, smooth=smooth)
                thresh_color = 'black'
            ax.axvline(peak_freq, color=color, linestyle='--', linewidth=2)
            ax.axvline(thresh_freq, color=thresh_color, linestyle='-.',
                       linewidth=2)
            legend += ['Peak Frequency', '80% Bandwidth']

        # plot unweighted if smoothed
        if iso_weighted and smooth:
            plot_kw = {'color': 'black', 'linewidth': 2}
            legend += ['Smoothed Raw FFT']
            freq, amp, _, _ = self.calculate_frequency_spectrum(
                sig_name, sample_rate, cutoff=cutoff, iso_weighted=False,
                smooth=smooth)
            ax = plot_frequency_spectrum(freq, amp, ax=ax, plot_kw=plot_kw)

        # plot smooth (and possible weighted)
        if smooth:
            freq, amp, _, _ = self.calculate_frequency_spectrum(
                sig_name, sample_rate, cutoff=cutoff,
                iso_weighted=iso_weighted, smooth=smooth)
            if iso_weighted:
                color = 'C0'
            else:
                color = 'black'
            plot_kw = {'color': color, 'linewidth': 2}
            legend += ['Smoothed ' +
                       ('ISO Weighted FFT' if iso_weighted else '')]
        else:
            plot_kw = {'color': 'gray', 'alpha': 0.8}

        ax = plot_frequency_spectrum(freq, amp, ax=ax, plot_kw=plot_kw)

        if cutoff is not None:
            ax.set_xlim((0.0, cutoff))

        ax.legend(legend)

        return ax

    def plot_signal(self, sig_name, show_rms=False, show_vdv=False, ax=None):
        """Returns a plot of the signal linearly interpolated at all sampled
        time values across all sensors.

        Parameters
        ==========
        sig_name : string
            Column name to down sample.
        show_rms : boolean
            Plots horizontal lines about the mean representing the root mean
            square.
        show_vdv : boolean
            Plots horizontal lines about the mean representing the vibration
            dose value.
        ax : Axis
            Matplotlib axis to plot on.

        """
        interped = self.imu_data[sig_name].interpolate(method='time')
        ax = interped.plot(ax=ax)
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        if 'acc' in sig_name.lower():
            too_big = interped[interped > 16.0*9.81]
            too_small= interped[interped < -16.0*9.81]
            ax = too_big.plot(ax=ax, linestyle='', marker='o', color='red')
            ax = too_small.plot(ax=ax, linestyle='', marker='o', color='red')
        if show_rms or show_vdv:
            mean = self.imu_data[sig_name].mean()
        if show_rms:
            rms = self.calc_rms(sig_name)
            ax.axhline(mean + rms, color='black')
            ax.axhline(mean - rms, color='black')
        dur_string = 'Duration: {:1.1f}s'.format(self.calc_duration())
        if show_vdv:
            vdv = self.calc_vdv(sig_name)
            dur_string += f', VDV = {vdv:0.2f}'
        ax.figure.text(0.01, 0.01, dur_string)
        # TODO : Not the case if gyro signal is selected, e.g.
        ax.set_ylabel('Acceleration\n[m/s$^2$]')
        ax.set_xlabel('Time [HH:MM:SS]')
        ax.grid()
        return ax

    def calc_rms(self, sig_name):
        """Returns the RMS of the raw signal data."""
        mean_subtracted = (self.imu_data[sig_name] -
                           self.imu_data[sig_name].mean())
        return np.sqrt(np.mean(mean_subtracted**2))

    def calc_max_peak(self, sig_name):
        """Returns the MAX value of the shock events."""
        max_peak_sh = abs(self.imu_data[sig_name]).max()
        return max_peak_sh

    @functools.cache
    def calc_crest_factor(self, sig_name, sample_rate, cutoff=None):
        """Returns the crest factor: ratio of maximum absolute value to the RMS
        (no ISO weighting).

        Notes
        =====
        ISO 2631-1 says that if the crest factor is greater than 9, then other
        methods should be used for assessment, like VDV.

        """
        _, x = self.down_sample(sig_name, sample_rate, cutoff=cutoff)
        rms = self.calc_spectrum_rms(sig_name, sample_rate, cutoff=cutoff)
        return np.max(np.abs(x))/rms

    @functools.cache
    def calc_spectrum_rms(self, sig_name, sample_rate, cutoff=None,
                          iso_weighted=False):
        """Returns the root mean square of the signal, calculated from the
        frequency spectrum which takes into account the time series filtering
        and ISO 2631-1 weights.

        Parameters
        ==========
        sig_name : string
            Column name to down sample.
        sample_rate : integer
            Sample rate in Hertz.
        cutoff : None or float
            If a float is supplied the signal will be downsampled and low pass
            filtered at the provided cutoff frequency in Hz before the FFT is
            applied.
        iso_weighted : boolean
            If true, spectrum will be weighted using the filters in ISO 2631-1.

        Returns
        =======
        float
            RMS of the signal (or ISO weighted signal).

        """

        # NOTE : I tried many times to calculate the RMS from the result of
        # dtk.process.freq_spectrum() but the value was always off. The reason
        # is that the nextpow2() calculation does something that causes a
        # direct application of Parseval's theorem to be incorrect. So I have
        # to calculate the FFT here without normalization and without the
        # nextpow2() to get the correct RMS value. If the normalization is
        # applied this can work:
        # X = fft(x, norm='forward')
        # RMS = np.sqrt(np.sum(np.abs(X*len(x))**2))/np.sqrt(len(X)**2)

        t, x = self.down_sample(sig_name, sample_rate, cutoff=cutoff)

        sample_time = 1.0/sample_rate

        X = np.fft.fft(x)
        f = np.fft.fftfreq(len(X), d=sample_time)
        # take right half spectrum and remove dc component (first value)
        power = X[1:len(X)//2]
        freq = f[1:len(X)//2]

        if iso_weighted:
            table_freq = iso_filter_df_01.index.values
            if 'acc' in sig_name:
                if 'acc_ver' in sig_name:
                    col = 'vertical_acceleration_z'
                else:
                    col = 'translational_acceleration_xy'
            elif 'gyr' in sig_name:
                col = 'rotation_speed_xyz'
            else:
                raise ValueError('no weights!')
            table_weights = iso_filter_df_01[col].values/1000.0
            weights = np.interp(freq, table_freq, table_weights)
            power = weights*power

        return np.sqrt(2*np.sum(np.abs(power)**2))/np.sqrt(len(X)**2)

    @functools.cache
    def calc_magnitude_rms(self, signal_prefix, sample_rate, cutoff=None,
                           iso_weighted=False):
        """Returns the RMS of the magnitude of a signal.

        Parameters
        ==========
        signal_prefix : string
            Example: ``SeatBotacc`` or ``SeatBotgyr`` or ``SeatHeadacc``
        cutoff : None or float
            If a float is supplied the signal will be downsampled and low pass
            filtered at the provided cutoff frequency in Hz before the FFT is
            applied.
        iso_weighted : boolean
            If true, spectrum will be weighted using the filters in ISO 2631-1.

        Notes
        =====

        We already calculate vector magnitudes but if we want to apply ISO
        weights we have to calculate the vector magnitude after the weights are
        applied.

        ``RMS = sqrt( kx^2 awx^2 + ky^2 awy^2 + kz^2 awz^2)``

        """
        ver_rms = self.calc_spectrum_rms(signal_prefix + '_ver', sample_rate,
                                         cutoff=cutoff,
                                         iso_weighted=iso_weighted)
        lat_rms = self.calc_spectrum_rms(signal_prefix + '_lat', sample_rate,
                                         cutoff=cutoff,
                                         iso_weighted=iso_weighted)
        lon_rms = self.calc_spectrum_rms(signal_prefix + '_lon', sample_rate,
                                         cutoff=cutoff,
                                         iso_weighted=iso_weighted)
        return np.sqrt(ver_rms**2 + lat_rms**2 + lon_rms**2)

    def calc_vdv(self, sig_name, duration=None):
        """Returns the VDV of the raw signal data.

        Parameters
        ==========
        sig_name : string
        duration : float, optional

        Returns
        =======
        vdv : float
            Returns NaN if duration is larger than total signal time duration.

        Notes
        =====

        Vibration Dose Value is the 4th root of the integral of the signal with
        respect to time.

        """
        fourthed = self.imu_data[sig_name].dropna()**4
        time = datetime2seconds(fourthed.index)
        if duration is not None:
            if duration > time[-1]:
                return np.nan
            max_time_idx = np.argmin(np.abs(time - duration)) - 1
            fourthed = fourthed.values[:max_time_idx]
            time = time[:max_time_idx]
        return trapezoid(fourthed, x=time)**(1/4)

    def calc_duration(self):
        """Returns the total duration of the trial in seconds."""
        return datetime2seconds(self.imu_data.index)[-1]

    def calc_speed_stats(self):
        """Returns the mean and standard deviation of the speed during the
        trial."""
        return self.imu_data['Speed'].mean(), self.imu_data['Speed'].std()

    @functools.cache
    def calc_spectrum_features(self, sig_name, sample_rate, cutoff=None,
                               iso_weighted=False, smooth=False):
        """Returns features of the frequency spectrum.

        Parameters
        ==========
        sig_name : string
            Column name to down sample.
        sample_rate : integer
            Sample rate in Hertz.
        iso_weighted : boolean
            If true, spectrum will be weighted using the filters in ISO 2631-1.
        smooth : boolean
            If true, the spectrum will be filtered using a 2nd order zero-lag
            Butterworth filter.

        Returns
        =======
        max_amp : float
            Maximum amplitude in the spectrum.
        peak_freq : float
            Frequency in Hertz at the maximum amplitude in the spectrum.
        thresh_freq : float
            Frequency at 80% of the area under the spectrum.

        """
        freq, amp, _, _ = self.calculate_frequency_spectrum(
            sig_name, sample_rate, cutoff=cutoff, iso_weighted=iso_weighted,
            smooth=smooth)
        max_amp = np.max(amp)
        peak_freq = freq[np.argmax(amp)]
        area = cumulative_trapezoid(amp, freq)
        threshold = 0.8*area[-1]
        idx = np.argwhere(area < threshold)[-1, 0]
        thresh_freq = freq[idx]
        return max_amp, peak_freq, thresh_freq


class Session():
    """Represents a continous period of data collection from multiple Shimmer
    IMUs, called a "session".

    Parameters
    ==========
    session_label : string
        ``sessionXXX`` where ``XXX`` is a three digit number ``000``, ``001``,
        etc. These labels are defined in ``data/session.yml``.

    """
    sensor_labels = [
        'BotTrike',
        'FrontWheel',
        'RearWheel',
        'SeatBot',
        'SeatHead',
    ]
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

    def __str__(self):
        desc = header('Session: ' + self.session_label, sym='-')
        desc += (f"\nVehicle: {self.meta_data['vehicle_type']} "
                 f"{self.meta_data['vehicle']}: "
                 f"{self.meta_data['brand']} {self.meta_data['model']}")
        desc += (f"\nSeat: {self.meta_data['seat']} with "
                 f"{self.meta_data['baby_age']} month, "
                 f"{self.meta_data['baby_mass']} kg baby")

        if hasattr(self, 'trial_bounds'):
            trial_info = {k: len(v) for k, v in self.trial_bounds.items()}
            desc += f"\nTrials: {trial_info}"
        else:
            desc += "\nTrials: no trials available"

        if hasattr(self, 'imu_data'):
            desc += (f"\nTotal duration: "
                     f"{datetime2seconds(self.imu_data.index)[-1]:0.2f}")
            desc += (f"\nMain data frame memory usage: "
                     f"{self.imu_data.memory_usage().sum()/1e6:0.2f} megabytes")

        return desc

    def _repr_html_(self):
        return '<pre>' + self.__str__() + '</pre>'

    def load_data(self):
        """Loads the IMU CSV files for this session into ``imu_data_frames``
        and the trial bound data into ``bounds_data_frame``."""

        self.imu_data_frames = load_session_files(self.session_label)

        self.trial_bounds = {}
        if self.meta_data['trial_bounds_file'] is None:
            self.bounds_data_frame = None
        else:
            path_to_bounds_file = os.path.join(
                PATH_TO_DATA_DIR, 'Interval_indexes',
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

    def print_memory_usage(self):
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

        self.imu_data = merge_imu_data_frames(
            *self.imu_data_frames.values()).astype(
                pd.SparseDtype("float", np.nan))

        if minimize_memory:
            # save memory by deleting the original data frames
            del self.imu_data_frames
            gc.collect()
            self.imu_data_frames = {}

    def extract_trial(self, trial_name, trial_number=0, split=None):
        """Selects a trial from ``imu_data`` based on the manually defined
        bounds stored in ``bounds_data_frame``.

        Parameters
        ==========
        trial_name : string
            Examples are ``static``, ``aula``, ``pave``, ``klinkers``,
            ``stoeptegels``, etc.
        trial_number : integer
            More than one trial with the same name may be present. Use this
            value to select the first, second, third, instance of that trial.
        split : None or float
            If set to a duration in seconds, the trial will be split up into
            multiple trials.

        Returns
        =======
        DataFrame or list of DataFrame
            Slice(s) of ``imu_data``.

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

        trial_df = self.imu_data[start_idx:stop_idx].apply(to_dense)

        start = trial_df.index[0]
        stop = trial_df.index[-1]

        if split is not None:
            duration = (stop - start).total_seconds()
            if duration/split < 1.0:  # don't split
                return self.imu_data[start_idx:stop_idx].apply(to_dense)
            else:
                splits = []
                split_idxs = list(range(int(duration//split)))
                for i in split_idxs:
                    t0 = start + datetime.timedelta(seconds=split*i)
                    if i == split_idxs[-1]:
                        tf = stop
                    else:
                        tf = start + datetime.timedelta(
                            seconds=split*(i + 1))
                    splits.append(self.imu_data[t0:tf].apply(to_dense))
                return splits

        return trial_df

    def rotate_imu_data(self, subtract_gravity=True):
        """Adds new columns to the ``imu_data`` data frame in which the
        accelerometer and rate gyro axes are rotated about the vehicle's
        lateral axis aligning one axis with the vertical (direction of gravity)
        and one with longitudinal.

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

    def plot_speed_with_trial_bounds(self):
        """Createas a plot of forward speed versus time for the whole session
        with shaded labeled areas for each trial."""
        fig, ax = plt.subplots(figsize=(16, 3), layout='constrained')
        ax = self.imu_data['Speed_kph'].dropna().plot(ax=ax, linestyle='',
                                                      marker='.',
                                                      x_compat=True)
        ax.xaxis.set_major_locator(mdates.MinuteLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        for idx, row in self.bounds_data_frame.iterrows():
            start, end = row['start_time'], row['end_time']
            try:
                chunk = self.imu_data.loc[start:end,
                                          'Speed_kph'].sparse.to_dense()
            except AttributeError:
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


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Process One Session and Trial')
    parser.add_argument('-s', '--session', type=str, default='session001')
    parser.add_argument('-t', '--trial', type=str, default='aula')
    parser.add_argument('-r', '--sample_rate', type=int, default=400)
    parser.add_argument('-p', '--plot', action='store_true')
    a = parser.parse_args()

    session_label = a.session
    trial_label = a.trial
    sample_rate = a.sample_rate
    plot = a.plot

    if plot:
        plot_iso_weights(iso_filter_df_01, iso_filter_df_02)

    s = Session(session_label)
    s.load_data()
    s.merge_imu_data()
    s.rotate_imu_data(subtract_gravity=False)
    s.calculate_travel_speed()
    s.calculate_vector_magnitudes()

    print(s)

    if plot:
        s.plot_accelerometer_rotation()

    s.rotate_imu_data()
    if 'synchro' in s.trial_bounds and plot:
        s.plot_time_sync()

    if plot:
        s.plot_speed_with_trial_bounds()
        s.plot_raw_time_series(trial=trial_label, gyr=False)
        s.plot_raw_time_series(trial=trial_label, acc=False)

    tr = Trial(s.meta_data, s.extract_trial(trial_label))
    if plot:
        # TODO : For some reason, this plots on the last axes of the
        # iso_weights plot if the figure creation is not first.
        fig, ax = plt.subplots(layout='constrained', figsize=(8, 2))
        tr.plot_signal("SeatBotacc_ver", show_rms=True, show_vdv=True, ax=ax)

    freq, amp, tim, sig = tr.calculate_frequency_spectrum('SeatBotacc_ver',
                                                          sample_rate)
    rms_time = np.sqrt(np.mean(sig**2))
    rms_spec = np.sqrt(0.5*np.sum(amp**2))
    print('RMS from all time series data: ', tr.calc_rms("SeatBotacc_ver"))
    print('RMS from down sampled time series data: ', rms_time)
    print('RMS from amplitude spectrum (incorrect): ', rms_spec)
    print('RMS from power spectrum',
          tr.calc_spectrum_rms("SeatBotacc_ver", sample_rate))
    print('RMS from power spectrum (iso weighted)',
          tr.calc_spectrum_rms("SeatBotacc_ver", sample_rate,
                               iso_weighted=True))
    print("RMS of the vector magnitude (from spectrum and ISO weighted): ",
          tr.calc_magnitude_rms("SeatBotacc", sample_rate,
                                iso_weighted=True))
    print("Crest factor: ",
          tr.calc_crest_factor("SeatBotacc_ver", sample_rate))
    print("VDV: ", tr.calc_vdv("SeatBotacc_ver"))

    if plot:
        tr.plot_frequency_spectrum('SeatBotacc_ver', sample_rate, smooth=True,
                                   iso_weighted=True,
                                   show_features='bandwidth from smooth')

        plt.show()
