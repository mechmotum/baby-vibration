import os

import yaml
import pandas as pd
import numpy as np
from dtk.inertia import x_rot, y_rot, z_rot

from paths import PATH_TO_SESSION_DATA, PATH_TO_REPO


def header(msg, sym='*'):
    lines = [sym*2*len(msg)]
    lines.append(
        '|' +
        ' '*(len(msg)//2 - 1) +
        msg +
        ' '*(len(msg)//2 + (-1 if len(msg) % 2 == 0 else -0)) +
        '|'
    )
    lines.append(sym*2*len(msg))
    return '\n'.join(lines)


def print_header(msg, sym='*'):
    print(header(msg, sym=sym))


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
# The code below this line was taken from the row_filter source code which is
# licensed with the MIT license (but I also authored it, so I own the
# copyright):
#
# https://gitlab.com/mechmotum/row_filter/-/blob/master/row_filter/complementary.py


def realtime_filter(t, x, cutoff, typ='low'):
    """Returns a 2nd order low- or high-pass filtered signal. Executes a
    realtime capable filter as a batch process.

    Parameters
    ==========
    t : array_like, shape(n,)
        The monotonically increasing time values in seconds. The sample rate
        can be irregular.
    x : array_like, shape(n,)
        The time varying signal's values.
    cutoff : float
        Cutoff frequency in Hz.
    typ : string, optional
        Either ``low`` or ``high`` for the filter type. Defaults to low.

    Returns
    =======
    y : ndarray, shape(n,)
        The filtered time varying signal.

    """

    y = np.zeros_like(t)
    w = y.copy()

    if typ == 'low':
        rt_filter = lowpass_filter
        args = x, y, w
    elif typ == 'high':
        rt_filter = highpass_filter
        z = y.copy()
        args = x, w[1:], w, z[1:], z
    else:
        raise ValueError("'low' and 'high' are the only valid filter types.")

    for i, (ti, xi) in enumerate(zip(t[1:], x[1:])):
        dt = ti - t[i]
        res = rt_filter(cutoff, dt, xi, *(a[i] for a in args))
        if typ == 'low':
            y[i+1], w[i+1] = res
        else:
            y[i+1], w[i+1], z[i+1] = res

    return y


def lowpass_filter(cutoff_freq, sample_time, x0, x1, y1, yd1):
    """Returns a 2nd order low pass Butterworth filtered value provided the
    prior unfiltered and filtered values. This is suitable for real-time
    filtering.

    Parameters
    ==========
    cutoff_freq : float
        Desired low pass cutoff frequency in Hertz.
    sample_time : float
        Time between current sample and prior sample in seconds.
    x0 : float
        Current unfiltered value.
    x1 : float
        Prior unfiltered value.
    y1 : float
        Prior filtered value.
    yd1 : float
        Prior derivative of the filtered value.

    Returns
    =======
    y0 : float
        Current filtered signal value.
    yd0 :
        Current filtered signal derivative.

    Notes
    =====

    These are the counts of the operations in the code:

    +     : 9
    -     : 4
    *     : 27
    /     : 8
    **    : 5
    sqrt  : 1
    ----------
    total : 54

    """

    # Compute coefficients of the state equation
    a = (2 * np.pi * cutoff_freq)**2
    b = np.sqrt(2) * 2 * np.pi * cutoff_freq

    # Integrate the filter state equation using the midpoint Euler method with
    # time step h
    h = sample_time
    denom = 4 + 2*h*b + h**2 * a

    A = (4 + 2*h*b - h**2*a)/denom
    B = 4*h/denom
    C = -4*h*a/denom
    D = (4 - 2*h*b - h**2*a)/denom
    E = 2*h**2*a/denom
    F = 4*h*a/denom

    y0 = A * y1 + B * yd1 + E*(x0 + x1) / 2
    yd0 = C * y1 + D * yd1 + F*(x0 + x1) / 2

    return y0, yd0


def highpass_filter(cutoff_freq, sample_time, xi, xim1, z1i, z1im1, z2i,
                    z2im1):
    """Returns a 2nd order high pass Butterworth filtered value provided the
    prior unfiltered values. This is suitable for real-time filtering.

    Parameters
    ==========
    cutoff_freq : float
        Desired high pass cutoff frequency in Hertz.
    sample_time : float
        Time between current sample and prior sample in seconds.
    xi : float
        Current value of unfiltered signal.
    xim1 : float
        Prior value of unfiltered signal.
    z1im1: float
        Prior value of the first state.
    z2im1: float
        Prior value of the second state.

    Returns
    =======
    yi : float
        Current value of the filtered signal.
    z1i : float
        Current value of the first state.
    z21 : float
        Current value of the second state.

    Notes
    =====

    These are the counts of the operations in the code:

    +     : 10
    -     : 5
    *     : 23
    /     : 3
    **    : 2
    sqrt  : 1
    ----------
    total : 44

    """

    h = sample_time
    w0 = 2 * np.pi * cutoff_freq  # convert to radians

    a0 = np.sqrt(2)*h*w0
    a1 = h**2
    a2 = w0**2
    a3 = a1*a2
    a4 = 2*a0
    a5 = a3 + a4 + 4
    a6 = 1/a5
    a7 = a1*xi + a1*xim1 - a3*z2im1 + a4*z2im1 + 4*h*z1im1 + 4*z2im1
    a8 = a2*h

    z1i = a6*(a5*(-a0*z1im1 - a8*z2im1 + h*xi + h*xim1 + 2*z1im1) -
              a7*a8)/(a0 + 2)
    z2i = a6*a7
    yi = (z1i - z1im1) / h

    return yi, z1i, z2i
