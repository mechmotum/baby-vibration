from collections import defaultdict
import argparse
import gc
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from data import Session, Trial, iso_filter_df_01, iso_filter_df_02
from functions import print_header
from plots import plot_iso_weights
from html_templates import IMG, H2, H3, P
from paths import (PATH_TO_DATA_DIR, PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR,
                   PATH_TO_TIME_HIST_DIR, PATH_TO_SPECT_DIR,
                   PATH_TO_ACCROT_DIR, PATH_TO_SYNC_DIR)


def process_sessions(start_num, end_num, signal, sample_rate):

    print('Processing {} to {}'.format(start_num, end_num))

    if start_num > 0:
        print_header('Loading existing pickle files', sym='*')
        with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'rb') as f:
            html_data = pickle.load(f)
        with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'rb') as f:
            stats_data = pickle.load(f)
    else:
        print_header('Creating new pickle files', sym='*')
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
        'shock': 'Shock',
        'stoeptegels': 'Sidewalk Pavers',
        'tarmac': 'Tarmac',
        'shock': 'Shock',
    }

    if end_num == 99:
        end_num = None
    sessions_to_process = session_labels[start_num:end_num]

    for session_label in sessions_to_process:
        print_header('Loading: {}'.format(session_label), sym='=')
        s = Session(session_label)
        html_data['sess_html'].append(H2.format(session_label))
        html_data['spec_html'].append(H2.format(session_label))
        html_data['srot_html'].append(H2.format(session_label))
        html_data['trial_html'].append(H2.format(session_label))
        html_data['sync_html'].append(H2.format(session_label))
        if s.meta_data['trial_bounds_file'] is None:
            print('Missing bounds file, skipping:', session_label)
            html_data['sess_html'].append(P.format(
                'missing bounds, skipped: ' + session_label))
            html_data['spec_html'].append(P.format(
                'missing bounds, skipped: ' + session_label))
            html_data['srot_html'].append(P.format(
                'missing bounds, skipped: ' + session_label))
            html_data['trial_html'].append(P.format(
                'missing bounds, skipped: ' + session_label))
            html_data['sync_html'].append(P.format(
                'missing bounds, skipped: ' + session_label))
        else:
            s.load_data()
            s.rotate_imu_data(subtract_gravity=False)
            axes = s.plot_accelerometer_rotation()
            axes[0, 0].figure.savefig(os.path.join(PATH_TO_ACCROT_DIR,
                                                   session_label + '.png'),
                                      dpi=300)
            plt.clf()
            html_data['srot_html'].append(
                IMG.format('accrot', session_label + '.png'))
            s.rotate_imu_data()
            s.calculate_travel_speed(smooth=True)
            s.calculate_vector_magnitudes()

            print(s)  # print after full DataFrame is constructed
            html_data['sess_html'].append(s._repr_html_())
            html_data['spec_html'].append(s._repr_html_())
            html_data['srot_html'].append(s._repr_html_())
            html_data['trial_html'].append(s._repr_html_())
            html_data['sync_html'].append(s._repr_html_())

            ax = s.plot_speed_with_trial_bounds()
            ses_img_fn = session_label + '.png'
            ax.figure.savefig(os.path.join(PATH_TO_BOUNDS_DIR, ses_img_fn),
                              dpi=300)
            plt.clf()
            html_data['sess_html'].append(IMG.format('bounds', ses_img_fn))

            if 'synchro' in s.trial_bounds:
                ax = s.plot_time_sync()
                sync_img_fn = session_label + '.png'
                ax.figure.savefig(os.path.join(PATH_TO_SYNC_DIR, sync_img_fn),
                                  dpi=300)
                plt.clf()
                html_data['sync_html'].append(IMG.format('sync', ses_img_fn))
            else:
                msg = 'no synchro trial, skipped: ' + session_label
                html_data['sync_html'].append(P.format(msg))

            plt.close('all')

            for mot_trial in motion_trials.keys():
                if mot_trial in s.trial_bounds:
                    if mot_trial != 'shock':
                        html_data['trial_html'].append(H3.format(mot_trial))
                        html_data['spec_html'].append(H3.format(mot_trial))
                    else:
                        html_data['trial_html'].append(H3.format(mot_trial))
                    for trial_num in s.trial_bounds[mot_trial]:
                        print('Trial Surface and Number: ', mot_trial,
                              trial_num)

                        # TODO : Make extract_trial produce a Trial
                        dfs = s.extract_trial(mot_trial,
                                              trial_number=trial_num,
                                              split=20.0)
                        if isinstance(dfs, pd.DataFrame):
                            dfs = [dfs]

                        print('{} repetitions in this trial.'.format(len(dfs)))

                        for rep_num, df in enumerate(dfs):

                            file_name = '-'.join([
                                session_label,
                                't' + str(trial_num),
                                mot_trial,
                                s.meta_data['vehicle_type'],
                                s.meta_data['vehicle'],
                                s.meta_data['seat'],
                                str(s.meta_data['baby_age']),
                                signal,
                                'rep' + str(rep_num),
                            ])

                            md = s.meta_data.copy()
                            md['surface'] = mot_trial
                            md['trial_number'] = trial_num
                            md['repetition_number'] = rep_num
                            trial = Trial(md, df)

                            # TODO : RMS and VDV are calculated from
                            # oversampled data, should I change to calculate
                            # from the downsampled data?
                            max_amp_shock = np.nan
                            rms = trial.calc_rms(signal)
                            vdv = trial.calc_vdv(signal)
                            # TODO : Move this to the input of process()
                            cutoff = 120.0  # Hz
                            rms_iso = trial.calc_spectrum_rms(
                                signal, sample_rate, cutoff=cutoff,
                                iso_weighted=True)
                            rms_mag_iso = trial.calc_magnitude_rms(
                                signal.split('_')[0], sample_rate,
                                cutoff=cutoff, iso_weighted=True)
                            crest_factor = trial.calc_crest_factor(
                                signal, sample_rate, cutoff=cutoff)
                            duration = trial.calc_duration()
                            avg_speed, std_speed = trial.calc_speed_stats()
                            max_amp, peak_freq, thresh_freq = \
                                trial.calc_spectrum_features(
                                    signal, sample_rate, cutoff=cutoff,
                                    smooth=True, iso_weighted=True)
                            max_amp_shock = trial.calc_max_peak(signal)

                            stats_data[signal + '_rms'].append(rms)
                            stats_data[signal + '_rms_iso'].append(rms_iso)
                            stats_data[signal.split('_')[0] +
                                       '_rms_mag_iso'].append(rms_mag_iso)
                            stats_data[signal + '_vdv'].append(vdv)
                            stats_data['Crest Factor'].append(crest_factor)
                            stats_data['Duration [s]'].append(duration)
                            stats_data['Mean Speed [m/s]'].append(avg_speed)
                            stats_data['STD DEV of Speed [m/s]'].append(
                                std_speed)
                            stats_data['Max Spectrum Amp [m/s/s]'].append(
                                max_amp)
                            stats_data['Peak Frequency [Hz]'].append(peak_freq)
                            stats_data['Bandwidth [Hz]'].append(
                                thresh_freq)
                            stats_data['Peak Value [m/s/s]'].append(max_amp_shock)
                            stats_data['Baby Age [mo]'].append(
                                s.meta_data['baby_age'])
                            stats_data['Baby Mass [kg]'].append(
                                s.meta_data['baby_mass'])
                            stats_data['Seat'].append(s.meta_data['seat'])
                            stats_data['Session'].append(session_label[-3:])
                            stats_data['Road Surface'].append(
                                motion_trials[mot_trial])
                            stats_data['Trial Repetition'].append(trial_num)
                            stats_data['Vehicle'].append(
                                s.meta_data['vehicle'])
                            stats_data['Vehicle Type'].append(
                                s.meta_data['vehicle_type'])
                            stats_data['repetition_number'].append(rep_num)

                            fig, ax = plt.subplots(layout='constrained',
                                                   figsize=(8, 2))
                            if mot_trial == 'shock':  # no RMS and VDV
                                ax = trial.plot_signal(signal, ax=ax)
                            else:
                                ax = trial.plot_signal(signal, show_rms=True,
                                                       show_vdv=True, ax=ax)
                            ax.figure.savefig(os.path.join(
                                PATH_TO_TIME_HIST_DIR, file_name + '.png'),
                                dpi=300)
                            plt.clf()
                            html_data['trial_html'].append(
                                IMG.format('time_hist', file_name + '.png'))

                            ax = trial.plot_frequency_spectrum(
                                signal, sample_rate, cutoff=cutoff,
                                smooth=True, iso_weighted=True,
                                show_features=True)
                            ax.set_title(file_name)
                            ax.figure.savefig(os.path.join(PATH_TO_SPECT_DIR,
                                                           file_name + '.png'),
                                              dpi=300)
                            plt.clf()
                            html_data['spec_html'].append(
                                IMG.format('spectrums', file_name + '.png'))

                            del trial
                            plt.close('all')
                        del dfs  # critical as this seems to be a copy!
                        gc.collect()
                else:
                    html_data['trial_html'].append(H3.format(mot_trial))
                    html_data['trial_html'].append(P.format(mot_trial +
                                                            ' not in session'))
                    html_data['spec_html'].append(H3.format(mot_trial))
                    html_data['spec_html'].append(P.format(mot_trial +
                                                           ' not in session'))

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

    ax1, ax2 = plot_iso_weights(iso_filter_df_01, iso_filter_df_02)
    ax1[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                       'iso-filter-weights-01.png'), dpi=300)
    ax2[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                       'iso-filter-weights-02.png'), dpi=300)
    plt.clf()

    process_sessions(a.start_num, a.end_num, a.signal,
                     a.sample_rate)
