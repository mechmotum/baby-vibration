import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data import Session, plot_frequency_spectrum, datetime2seconds
from data import PATH_TO_DATA_DIR, PATH_TO_FIG_DIR

PATH_TO_BOUNDS_DIR = os.path.join(PATH_TO_FIG_DIR, 'bounds')

SAMPLE_RATE = 200  # down sample data to this rate

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
        print(s.trial_bounds)
        s.rotate_imu_data()
        s.calculate_travel_speed()
        s.calculate_vector_magnitudes()
        if not os.path.exists(PATH_TO_BOUNDS_DIR):
            os.mkdir(PATH_TO_BOUNDS_DIR)
        ax = s.plot_speed_with_trial_bounds()
        ax.figure.savefig(os.path.join(PATH_TO_BOUNDS_DIR,
                                       session_label + '.png'))
        # TODO : No need to plot this same thing for every session.
        ax1, ax2 = s.plot_iso_weights()
        ax1[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                           'iso-filter-weights-01.png'))
        ax2[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                           'iso-filter-weights-02.png'))
        plt.close('all')
        for mot_trial in motion_trials:
            if mot_trial in s.trial_bounds:
                count += 1
                for trial_num in s.trial_bounds[mot_trial]:
                    stats_data['surface'].append(mot_trial.lower())
                    stats_data['vehicle'].append(s.meta_data['vehicle'])
                    stats_data['vehicle_type'].append(
                        s.meta_data['vehicle_type'])
                    stats_data['baby_age'].append(s.meta_data['baby_age'])

                    df = s.extract_trial(mot_trial, trial_number=trial_num)
                    dur = datetime2seconds(df.index)[-1]
                    stats_data['duration'].append(dur)
                    stats_data['speed_avg'].append(df['Speed'].mean())
                    stats_data['speed_std'].append(df['Speed'].std())

                    signal = 'SeatBotacc_ver'
                    freq, amp = s.calculate_frequency_spectrum(
                        signal, SAMPLE_RATE, trial=mot_trial,
                        iso_weighted=True)
                    # TODO : the factor 2 is because it is a two-sided FFT,
                    # double check that 2 is needed.
                    rms = np.sqrt(2.0*np.mean(amp**2))
                    stats_data['SeatBot_acc_ver_rms'].append(rms)
                    ax = plot_frequency_spectrum(freq, amp, rms, SAMPLE_RATE)
                    file_name = '-'.join([
                        str(count),
                        stats_data['surface'][-1],
                        stats_data['vehicle'][-1],
                        stats_data['vehicle_type'][-1],
                        str(stats_data['baby_age'][-1]),
                        signal,
                    ])
                    ax.set_title(file_name)
                    spectrum_dir = os.path.join(PATH_TO_FIG_DIR, 'spectrums')
                    if not os.path.exists(spectrum_dir):
                        os.mkdir(spectrum_dir)
                    ax.figure.savefig(os.path.join(spectrum_dir,
                                                   file_name + '.png'))

                    plt.close('all')
                    del df  # critical as this seems to be a copy!
    del s

stats_df = pd.DataFrame(stats_data)
print(stats_df)
groups = ['vehicle', 'baby_age', 'surface']
print(stats_df.groupby(groups)['SeatBot_acc_ver_rms'].mean())
# way to may box plot comparisons
#stats_df.groupby('surface').boxplot(subplots=False, column='SeatBot_acc_ver_rms')