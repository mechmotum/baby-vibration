import os
from collections import defaultdict
import gc
import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml

from run import (PATH_TO_DATA_DIR, PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR,
                 PATH_TO_TIME_HIST_DIR, PATH_TO_SPECT_DIR, PATH_TO_ACCROT_DIR)
from run import SAMPLE_RATE, SIGNAL, SIGNAL_RMS, START_SESSION, END_SESSION
from data import Session, plot_frequency_spectrum, datetime2seconds

if START_SESSION > 0:
    with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'rb') as f:
        html_data = pickle.load(f)
    with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'rb') as f:
        stats_data = pickle.load(f)
else:
    stats_data = defaultdict(list)
    html_data = {
        'sess_html': [],
        'trial_html': [],
        'spect_html': [],
        'srot_html': [],
    }
    if os.path.exists(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl')):
        os.remove(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'))
    if os.path.exists(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl')):
        os.remove(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'))

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

sessions_to_process = session_labels[START_SESSION:END_SESSION]

for sess_count, session_label in enumerate(sessions_to_process):
    print('Loading: ', session_label)
    s = Session(session_label)
    s.load_data()
    if not s.trial_bounds:
        print('Missing files, skipping:', session_label)
        del s
        gc.collect()
    else:
        print(s.trial_bounds)
        s.rotate_imu_data(subtract_gravity=False)
        axes = s.plot_accelerometer_rotation()
        axes[0, 0].figure.savefig(os.path.join(PATH_TO_ACCROT_DIR,
                                               session_label + '.png'))
        html_data['srot_html'].append('<h2>' + session_label + '</h2>')
        html_data['srot_html'].append('<img src="fig/accrot/' + session_label +
                                      '.png"></img>')
        s.rotate_imu_data()
        s.calculate_travel_speed()
        s.calculate_vector_magnitudes()

        ax = s.plot_speed_with_trial_bounds()
        ses_img_fn = session_label + '.png'
        ax.figure.savefig(os.path.join(PATH_TO_BOUNDS_DIR, ses_img_fn))
        html_data['sess_html'].append('<img src="fig/bounds/' + ses_img_fn +
                                      '"</img>')

        # TODO : No need to plot this same thing for every session, but need to
        # load a session for the data. Maybe disconnect this data from a
        # session.
        ax1, ax2 = s.plot_iso_weights()
        ax1[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                           'iso-filter-weights-01.png'))
        ax2[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                           'iso-filter-weights-02.png'))
        plt.clf()
        plt.close('all')
        del axes, ax, ax1, ax2
        gc.collect()

        html_data['trial_html'].append('<h2>{}</h2>'.format(session_label))
        html_data['spect_html'].append('<h2>{}</h2>'.format(session_label))
        for mot_trial in motion_trials:
            if mot_trial in s.trial_bounds:
                html_data['trial_html'].append('<h3>{}</h3>'.format(mot_trial))
                html_data['spect_html'].append('<h3>{}</h3>'.format(mot_trial))
                for trial_num in s.trial_bounds[mot_trial]:
                    print('Trial Surface and Number: ', mot_trial, trial_num)
                    stats_data['surface'].append(mot_trial.lower())
                    stats_data['vehicle'].append(s.meta_data['vehicle'])
                    stats_data['vehicle_type'].append(
                        s.meta_data['vehicle_type'])
                    stats_data['baby_age'].append(s.meta_data['baby_age'])

                    df = s.extract_trial(mot_trial, trial_number=trial_num)

                    file_name = '-'.join([
                        session_label,
                        's' + str(sess_count),
                        't' + str(trial_num),
                        stats_data['surface'][-1],
                        stats_data['vehicle'][-1],
                        stats_data['vehicle_type'][-1],
                        str(stats_data['baby_age'][-1]),
                        SIGNAL,
                    ])

                    fig, ax = plt.subplots(layout='constrained',
                                           figsize=(8, 2))
                    ax = df[SIGNAL].interpolate(method='time').plot(ax=ax)
                    ax.figure.savefig(os.path.join(PATH_TO_TIME_HIST_DIR,
                                                   file_name + '.png'))
                    html_data['trial_html'].append('<img src="fig/time_hist/' +
                                                   file_name + '.png"</img>')

                    dur = datetime2seconds(df.index)[-1]
                    stats_data['duration'].append(dur)
                    stats_data['speed_avg'].append(df['Speed'].mean())
                    stats_data['speed_std'].append(df['Speed'].std())

                    plt.clf()
                    plt.close('all')
                    del fig, ax
                    del df  # critical as this seems to be a copy!
                    gc.collect()

                    freq, amp = s.calculate_frequency_spectrum(
                        SIGNAL, SAMPLE_RATE, mot_trial, trial_number=trial_num)
                    rms = np.sqrt(2.0*np.mean(amp**2))
                    ax = plot_frequency_spectrum(freq, amp, rms, SAMPLE_RATE)

                    freq, amp = s.calculate_frequency_spectrum(
                        SIGNAL, SAMPLE_RATE, mot_trial, trial_number=trial_num,
                        iso_weighted=True)
                    rms = np.sqrt(2.0*np.mean(amp**2))
                    stats_data[SIGNAL_RMS].append(rms)
                    ax = plot_frequency_spectrum(freq, amp, rms, SAMPLE_RATE,
                                                 ax=ax)
                    ax.set_title(file_name)
                    ax.legend(['Unweighted', 'Unweighted RMS',
                               'Weighted', 'Weighted RMS'])
                    ax.figure.savefig(os.path.join(PATH_TO_SPECT_DIR,
                                                   file_name + '.png'))
                    html_data['spect_html'].append('<img src="fig/spectrums/' +
                                                   file_name + '.png"</img>')

                    plt.clf()
                    plt.close('all')
                    del freq, amp, rms, ax
                    gc.collect()
        del s
        gc.collect()

with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'wb') as f:
    pickle.dump(html_data, f)

with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'wb') as f:
    html_data = pickle.dump(stats_data, f)
