from collections import defaultdict
import argparse
import gc
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import cumulative_trapezoid

from data import Session
from functions import datetime2seconds
from html_templates import IMG, H2, H3, P
from paths import (PATH_TO_DATA_DIR, PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR,
                   PATH_TO_TIME_HIST_DIR, PATH_TO_SPECT_DIR,
                   PATH_TO_ACCROT_DIR, PATH_TO_SYNC_DIR)
from plots import plot_frequency_spectrum


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
        html_data = defaultdict(list)
        for pkl_file in ['html-data.pkl', 'stats-data.pkl']:
            if os.path.exists(os.path.join(PATH_TO_DATA_DIR, pkl_file)):
                os.remove(os.path.join(PATH_TO_DATA_DIR, pkl_file))

    with open(os.path.join(PATH_TO_DATA_DIR, 'sessions.yml')) as f:
        session_meta_data = yaml.safe_load(f)

    session_labels = list(session_meta_data.keys())

    motion_trials = {
        'aula': 'Sidewalk Slabs',
        'klinkers': 'Paver Bricks',
        'pave': 'Cobblestones',
        'stoeptegels': 'Sidewalk Pavers',
        'tarmac': 'Tarmac',
    }

    if end_num == 99:
        end_num = None
    sessions_to_process = session_labels[start_num:end_num]

    for session_label in sessions_to_process:
        msg = 'Loading: {}'.format(session_label)
        print('='*len(msg))
        print(msg)
        print('='*len(msg))
        s = Session(session_label)
        html_data['sess_html'].append(H2.format(session_label))
        html_data['spec_html'].append(H2.format(session_label))
        html_data['srot_html'].append(H2.format(session_label))
        html_data['trial_html'].append(H2.format(session_label))
        html_data['sync_html'].append(H2.format(session_label))
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

            if 'synchro' in s.trial_bounds:
                ax = s.plot_time_sync()
                sync_img_fn = session_label + '.png'
                ax.figure.savefig(os.path.join(PATH_TO_SYNC_DIR, sync_img_fn))
                plt.clf()
                html_data['sync_html'].append(IMG.format('sync', ses_img_fn))
            else:
                html_data['sync_html'].append(P.format('skipped: ' +
                                                       session_label))

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

            for mot_trial in motion_trials.keys():
                if mot_trial in s.trial_bounds:
                    html_data['trial_html'].append(H3.format(mot_trial))
                    html_data['spec_html'].append(H3.format(mot_trial))
                    for trial_num in s.trial_bounds[mot_trial]:
                        print('Trial Surface and Number: ', mot_trial,
                              trial_num)
                        stats_data['Baby Age [mo]'].append(
                            s.meta_data['baby_age'])
                        stats_data['Baby Mass [kg]'].append(
                            s.meta_data['baby_mass'])
                        stats_data['Seat'].append(s.meta_data['seat'])
                        stats_data['Session'].append(session_label[-3:])
                        stats_data['Road Surface'].append(
                            motion_trials[mot_trial])
                        stats_data['Trial Repetition'].append(trial_num)
                        stats_data['Vehicle'].append(s.meta_data['vehicle'])
                        stats_data['Vehicle Type'].append(
                            s.meta_data['vehicle_type'])

                        file_name = '-'.join([
                            session_label,
                            't' + str(trial_num),
                            mot_trial,
                            s.meta_data['vehicle_type'],
                            s.meta_data['vehicle'],
                            s.meta_data['seat'],
                            str(s.meta_data['baby_age']),
                            signal,
                        ])

                        df = s.extract_trial(mot_trial, trial_number=trial_num)

                        fig, ax = plt.subplots(layout='constrained',
                                               figsize=(8, 2))
                        ax = df[signal].interpolate(method='time').plot(ax=ax)
                        # TODO : These two are calculated from oversampled
                        # data, change to calculate from the downsampled data.
                        rms = np.sqrt(np.mean(df[signal]**2))
                        vdv = np.mean(df[signal]**4)**(0.25)
                        ax.axhline(rms, color='black')
                        ax.axhline(-rms, color='black')
                        ax.axhline(vdv, color='grey')
                        ax.axhline(-vdv, color='grey')
                        ax.set_ylabel('Acceleration [m/s$^2$]')
                        ax.set_xlabel('Time [HH:MM:SS]')
                        ax.figure.set_layout_engine('constrained')  # twice?
                        ax.figure.savefig(os.path.join(PATH_TO_TIME_HIST_DIR,
                                                       file_name + '.png'))

                        plt.clf()
                        html_data['trial_html'].append(
                            IMG.format('time_hist', file_name + '.png'))

                        duration = datetime2seconds(df.index)[-1]
                        stats_data['Duration [s]'].append(duration)
                        stats_data['Mean Speed [m/s]'].append(
                            df['Speed'].mean())
                        stats_data['Standard Deviation of Speed [m/s]'].append(
                            df['Speed'].std())

                        plt.close('all')
                        del df  # critical as this seems to be a copy!
                        gc.collect()

                        freq, amp, _, sig = s.calculate_frequency_spectrum(
                            signal, sample_rate, mot_trial,
                            trial_number=trial_num)
                        rms = np.sqrt(np.mean(sig**2))
                        stats_data[signal + '_rms'].append(rms)
                        vdv = np.mean(sig**4)**(0.25)
                        stats_data[signal + '_vdv'].append(vdv)
                        ax = plot_frequency_spectrum(freq, amp,
                                                     plot_kw={'color': 'gray',
                                                              'alpha': 0.8})

                        freq, amp, _, sig = s.calculate_frequency_spectrum(
                            signal, sample_rate, mot_trial,
                            trial_number=trial_num, smooth=True)
                        ax = plot_frequency_spectrum(freq, amp, ax=ax,
                                                     plot_kw={'color': 'C0',
                                                              'linewidth': 3})
                        peak_freq = freq[np.argmax(amp)]
                        area = cumulative_trapezoid(amp, freq)
                        threshold = 0.8*area[-1]
                        idx = np.argwhere(area < threshold)[-1, 0]
                        thresh_freq = freq[idx]
                        stats_data['Peak Frequency [Hz]'].append(peak_freq)
                        stats_data['Threshold Frequency [Hz]'].append(
                            thresh_freq)
                        ax.axvline(peak_freq, color='C1', linewidth=3)
                        ax.axvline(thresh_freq, color='C2', linewidth=3)
                        ax.set_title(file_name)
                        ax.legend(['FFT', 'Smoothed FTT', 'Peak Frequency',
                                   'Threshold Frequency'])
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
