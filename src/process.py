import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data import Session, plot_frequency_spectrum

PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
PATH_TO_FIG_DIR = os.path.join(PATH_TO_REPO, 'fig')
PATH_TO_DATA_DIR = os.path.join(PATH_TO_REPO, 'data')
if not os.path.exists(PATH_TO_FIG_DIR):
    os.mkdir(PATH_TO_FIG_DIR)


with open(os.path.join(PATH_TO_DATA_DIR, 'sessions.yml')) as f:
    session_meta_data = yaml.safe_load(f)

session_labels = list(session_meta_data.keys())

motion_trials = [
    'Aula',
    'klinkers',
    'pave',
    'stoeptegels',
    'tarmac',
]

stats_data = defaultdict(list)

count = 0
for session_label in session_labels:
    print('Loading: ', session_label)
    s = Session(session_label)
    s.load_data()
    if not s.trial_bounds:
        print('Missing files, skipping:', session_label)
    else:
        s.rotate_imu_data()
        s.calculate_travel_speed()
        s.calculate_vector_magnitudes()
        for mot_trial in motion_trials:
            if mot_trial in s.trial_bounds:
                count += 1
                for trial_num in s.trial_bounds[mot_trial]:
                    df = s.extract_trial(mot_trial, trial_number=trial_num)
                    dur = (df.index[-1] - df.index[0])/pd.offsets.Second(1)
                    stats_data['duration'] = dur
                    stats_data['surface'].append(mot_trial)
                    stats_data['vehicle'].append(s.meta_data['vehicle'])
                    stats_data['vehicle_type'].append(s.meta_data['vehicle_type'])
                    stats_data['baby_age'].append(s.meta_data['baby_age'])
                    stats_data['speed_avg'].append(df['Speed'].mean())
                    stats_data['speed_std'].append(df['Speed'].std())
                    signal = 'SeatBotacc_mag'
                    freq, amp = s.calculate_frequency_spectrum(signal,
                                                               trial=mot_trial)
                    rms = np.sqrt(np.mean(amp**2))
                    stats_data['SeatBot_acc_mag_rms'].append(rms)
                    ax = plot_frequency_spectrum(freq, amp, rms)
                    file_name = '-'.join([
                        str(count),
                        stats_data['surface'][-1],
                        stats_data['vehicle'][-1],
                        stats_data['vehicle_type'][-1],
                        str(stats_data['baby_age'][-1]),
                        signal,
                    ])
                    ax.set_title(file_name)
                    ax.figure.savefig(os.path.join(PATH_TO_FIG_DIR, file_name +
                                                   '.png'))
                    plt.close('all')
                    del df  # critical as this seems to be a copy!
    del s

print(pd.DataFrame(stats_data))
print(df.groupby(['vehicle', 'surface'])['SeatBot_acc_mag_rms'].mean())
