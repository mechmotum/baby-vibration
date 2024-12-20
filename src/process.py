from collections import defaultdict
import argparse
import gc
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml

from paths import (PATH_TO_DATA_DIR, PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR,
                   PATH_TO_TIME_HIST_DIR, PATH_TO_SPECT_DIR,
                   PATH_TO_ACCROT_DIR)
from html_templates import IMG, H2, H3, P
from data import Session, plot_frequency_spectrum, datetime2seconds


def process_sessions(start_num, end_num, signal, sample_rate):

    if start_num > 0:
        print('Loading existing pickle files')
        with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'rb') as f:
            html_data = pickle.load(f)
        with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'rb') as f:
            stats_data = pickle.load(f)
    else:
        print('Creating new pickle files')
        stats_data = defaultdict(list)
        html_data = {
            'sess_html': [],
            'spec_html': [],
            'srot_html': [],
            'trial_html': [],
        }
        for pkl_file in ['html-data.pkl', 'stats-data.pkl']:
            if os.path.exists(os.path.join(PATH_TO_DATA_DIR, pkl_file)):
                os.remove(os.path.join(PATH_TO_DATA_DIR, pkl_file))

    with open(os.path.join(PATH_TO_DATA_DIR, 'sessions.yml')) as f:
        session_meta_data = yaml.safe_load(f)

    session_labels = list(session_meta_data.keys())

    motion_trials = [
        'aula',
        'klinkers',
        'pave',
        'stoeptegels',
        'tarmac',
    ]

    if end_num == 99:
        end_num = None
    sessions_to_process = session_labels[start_num:end_num]

    for sess_count, session_label in enumerate(sessions_to_process):
        msg = 'Loading: {}'.format(session_label)
        print('='*len(msg))
        print(msg)
        print('='*len(msg))
        s = Session(session_label)
        html_data['sess_html'].append(H2.format(session_label))
        html_data['spec_html'].append(H2.format(session_label))
        html_data['srot_html'].append(H2.format(session_label))
        html_data['trial_html'].append(H2.format(session_label))
        if s.meta_data['trial_bounds_file'] is None:
            print('Missing files, skipping:', session_label)
            html_data['sess_html'].append(P.format('skipped: ' +
                                                   session_label))
            html_data['spec_html'].append(P.format('skipped: ' +
                                                   session_label))
            html_data['srot_html'].append(P.format('skipped: ' +
                                                   session_label))
            html_data['trial_html'].append(P.format('skipped: ' +
                                                    session_label))
            del s
        else:
            s.load_data()
            print(s.trial_bounds)
            s.rotate_imu_data(subtract_gravity=False)
            axes = s.plot_accelerometer_rotation()
            axes[0, 0].figure.savefig(os.path.join(PATH_TO_ACCROT_DIR,
                                                   session_label + '.png'))
            plt.clf()
            html_data['srot_html'].append(
                IMG.format('accrot', session_label + '.png'))
            s.rotate_imu_data()
            s.calculate_travel_speed(smooth=True)
            s.calculate_vector_magnitudes()

            ax = s.plot_speed_with_trial_bounds()
            ses_img_fn = session_label + '.png'
            ax.figure.savefig(os.path.join(PATH_TO_BOUNDS_DIR, ses_img_fn))
            plt.clf()
            html_data['sess_html'].append(IMG.format('bounds', ses_img_fn))

            # TODO : No need to plot this same thing for every session, but
            # need to load a session for the data. Maybe disconnect this data
            # from a session.
            ax1, ax2 = s.plot_iso_weights()
            ax1[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                               'iso-filter-weights-01.png'))
            ax2[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                               'iso-filter-weights-02.png'))
            plt.clf()

            plt.close('all')

            for mot_trial in motion_trials:
                if mot_trial in s.trial_bounds:
                    html_data['trial_html'].append(H3.format(mot_trial))
                    html_data['spec_html'].append(H3.format(mot_trial))
                    for trial_num in s.trial_bounds[mot_trial]:
                        print('Trial Surface and Number: ', mot_trial,
                              trial_num)
                        stats_data['session'].append(session_label[-3:])
                        stats_data['surface'].append(mot_trial.lower())
                        stats_data['surface_count'].append(trial_num)
                        stats_data['vehicle'].append(s.meta_data['vehicle'])
                        stats_data['vehicle_type'].append(
                            s.meta_data['vehicle_type'])
                        stats_data['baby_age'].append(s.meta_data['baby_age'])

                        file_name = '-'.join([
                            session_label,
                            's' + str(sess_count),
                            't' + str(trial_num),
                            stats_data['surface'][-1],
                            stats_data['vehicle'][-1],
                            stats_data['vehicle_type'][-1],
                            str(stats_data['baby_age'][-1]),
                            signal,
                        ])

                        df = s.extract_trial(mot_trial, trial_number=trial_num)

                        fig, ax = plt.subplots(layout='constrained',
                                               figsize=(8, 2))
                        ax = df[signal].interpolate(method='time').plot(ax=ax)
                        ax.figure.savefig(os.path.join(PATH_TO_TIME_HIST_DIR,
                                                       file_name + '.png'))

                        plt.clf()
                        html_data['trial_html'].append(
                            IMG.format('time_hist', file_name + '.png'))

                        duration = datetime2seconds(df.index)[-1]
                        stats_data['duration'].append(duration)
                        stats_data['speed_avg'].append(df['Speed'].mean())
                        stats_data['speed_std'].append(df['Speed'].std())

                        plt.close('all')
                        del df  # critical as this seems to be a copy!
                        gc.collect()

                        freq, amp, _, sig = s.calculate_frequency_spectrum(
                            signal, sample_rate, mot_trial,
                            trial_number=trial_num)
                        rms = np.sqrt(np.mean(sig**2))
                        ax = plot_frequency_spectrum(freq, amp, rms,
                                                     sample_rate)

                        freq, amp, _, sig = s.calculate_frequency_spectrum(
                            signal, sample_rate, mot_trial,
                            trial_number=trial_num, iso_weighted=True)
                        # TODO : this stores the unweighted RMS!
                        rms = np.sqrt(np.mean(sig**2))
                        stats_data[signal + '_rms'].append(rms)
                        ax = plot_frequency_spectrum(freq, amp, rms,
                                                     sample_rate, ax=ax)
                        ax.set_title(file_name)
                        ax.legend(['Unweighted', 'RMS', 'Weighted', 'RMS'])
                        ax.figure.savefig(os.path.join(PATH_TO_SPECT_DIR,
                                                       file_name + '.png'))

                        plt.clf()
                        html_data['spec_html'].append(
                            IMG.format('spectrums', file_name + '.png'))

                        plt.clf()

                        plt.close('all')
            del s
            gc.collect()

    with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'wb') as f:
        pickle.dump(html_data, f)

    with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'wb') as f:
        html_data = pickle.dump(stats_data, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process Sessions')
    parser.add_argument('start_num', type=int)
    parser.add_argument('end_num', type=int)
    parser.add_argument('signal', type=str)
    parser.add_argument('sample_rate', type=int)
    a = parser.parse_args()

    process_sessions(a.start_num, a.end_num, a.signal,
                     a.sample_rate)
