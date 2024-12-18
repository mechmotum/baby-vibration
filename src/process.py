import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data import Session, plot_frequency_spectrum, datetime2seconds
from data import PATH_TO_REPO, PATH_TO_DATA_DIR, PATH_TO_FIG_DIR

PATH_TO_BOUNDS_DIR = os.path.join(PATH_TO_FIG_DIR, 'bounds')
PATH_TO_TIME_HIST_DIR = os.path.join(PATH_TO_FIG_DIR, 'time_hist')
PATH_TO_SPECT_DIR = os.path.join(PATH_TO_FIG_DIR, 'spectrums')

NUM_SESSIONS = -1  # -1 for all
SAMPLE_RATE = 200  # down sample data to this rate

for dr in [PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR, PATH_TO_TIME_HIST_DIR,
           PATH_TO_SPECT_DIR]:
    if not os.path.exists(dr):
        os.mkdir(dr)

html_tmpl= """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Baby Vehicle Vibration Results</title>
  </head>
  <body>
  <h1>Mean RMS</h1>
{mean_table}
  <h1>Box Plots</h1>
{boxp_html}
  <h1>Sessions Segmented into Trials</h1>
{sess_html}
  <h1>ISO 2631-1 Weights</h1>
  <img src='fig/iso-filter-weights-01.png'</img>
  <img src='fig/iso-filter-weights-02.png'</img>
  <h1>Trials</h1>
{trial_table}
  <h1>Seat Pan Vertical Acceleration Spectrums</h1>
{spect_html}
  <h1>Seat Pan Vertical Acceleration Time Histories</h1>
{trial_html}
  </body>
</html>
"""

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

sess_html = []
trial_html = []
spect_html = []

for sess_count, session_label in enumerate(session_labels[:NUM_SESSIONS]):
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

        ax = s.plot_speed_with_trial_bounds()
        ses_img_fn = session_label + '.png'
        ax.figure.savefig(os.path.join(PATH_TO_BOUNDS_DIR, ses_img_fn))
        sess_html.append('<img src="fig/bounds/' + ses_img_fn + '"</img>')

        # TODO : No need to plot this same thing for every session, but need to
        # load a session for the data. Maybe disconnect this data from a
        # session.
        ax1, ax2 = s.plot_iso_weights()
        ax1[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                           'iso-filter-weights-01.png'))
        ax2[0].figure.savefig(os.path.join(PATH_TO_FIG_DIR,
                                           'iso-filter-weights-02.png'))
        plt.close('all')

        trial_html.append('<h2>{}</h2>'.format(session_label))
        spect_html.append('<h2>{}</h2>'.format(session_label))
        for mot_trial in motion_trials:
            if mot_trial in s.trial_bounds:
                trial_html.append('<h3>{}</h3>'.format(mot_trial))
                spect_html.append('<h3>{}</h3>'.format(mot_trial))
                for trial_num in s.trial_bounds[mot_trial]:
                    print('Trial #: ', trial_num)
                    stats_data['surface'].append(mot_trial.lower())
                    stats_data['vehicle'].append(s.meta_data['vehicle'])
                    stats_data['vehicle_type'].append(
                        s.meta_data['vehicle_type'])
                    stats_data['baby_age'].append(s.meta_data['baby_age'])

                    df = s.extract_trial(mot_trial, trial_number=trial_num)

                    signal = 'SeatBotacc_ver'

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

                    fig, ax = plt.subplots(layout='constrained',
                                           figsize=(8, 2))
                    ax = df[signal].interpolate(method='time').plot(ax=ax)
                    ax.figure.savefig(os.path.join(PATH_TO_TIME_HIST_DIR,
                                                   file_name + '.png'))
                    trial_html.append('<img src="fig/time_hist/' + file_name +
                                      '.png"</img>')

                    dur = datetime2seconds(df.index)[-1]
                    stats_data['duration'].append(dur)
                    stats_data['speed_avg'].append(df['Speed'].mean())
                    stats_data['speed_std'].append(df['Speed'].std())

                    freq, amp = s.calculate_frequency_spectrum(
                        signal, SAMPLE_RATE, mot_trial, trial_number=trial_num)
                    rms = np.sqrt(2.0*np.mean(amp**2))
                    ax = plot_frequency_spectrum(freq, amp, rms, SAMPLE_RATE)

                    freq, amp = s.calculate_frequency_spectrum(
                        signal, SAMPLE_RATE, mot_trial, trial_number=trial_num,
                        iso_weighted=True)
                    rms = np.sqrt(2.0*np.mean(amp**2))
                    stats_data['SeatBot_acc_ver_rms'].append(rms)
                    ax = plot_frequency_spectrum(freq, amp, rms, SAMPLE_RATE,
                                                 ax=ax)
                    ax.set_title(file_name)
                    ax.figure.savefig(os.path.join(PATH_TO_SPECT_DIR,
                                                   file_name + '.png'))
                    spect_html.append('<img src="fig/spectrums/' + file_name +
                                      '.png"</img>')

                    plt.close('all')
                    del df  # critical as this seems to be a copy!
    del s

stats_df = pd.DataFrame(stats_data)
print(stats_df)
groups = ['vehicle', 'baby_age', 'surface']
mean_df = stats_df.groupby(groups)['SeatBot_acc_ver_rms'].mean()
print(mean_df)

boxp_html = []

boxp_html.append('<h2>Speed</h2>')
fig, ax = plt.subplots(5, layout='constrained', figsize=(12, 8))
fig.suptitle('Speed Distributions')
stats_df.groupby('surface').boxplot(by=['vehicle', 'baby_age'],
                                    column='speed_avg', ax=ax)
fname = 'speed-dist-boxplot.png'
fig.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
boxp_html.append('<img src="fig/{}"></img>\n</br>'.format(fname))

for grp in ['surface', 'vehicle', 'vehicle_type', 'baby_age']:
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Speed Distribution Grouped By: {}'.format(grp))
    ax = stats_df.groupby(grp).boxplot(column='speed_avg', subplots=False,
                                       rot=45, ax=ax)
    fname = 'speed-by-{}-boxplot.png'.format(grp)
    fig.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
    boxp_html.append('<img src="fig/{}"></img>'.format(fname))

boxp_html.append('<h2>SeatBot_acc_ver</h2>')

fig, ax = plt.subplots(5, layout='constrained', figsize=(12, 8))
fig.suptitle('Seat Pan RMS Acceleration Distributions')
stats_df.groupby('surface').boxplot(by=['vehicle', 'baby_age'],
                                    column='SeatBot_acc_ver_rms', ax=ax)
fname = 'SeatBot_acc_ver-dist-boxplot.png'
fig.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
boxp_html.append('<img src="fig/{}"></img>\n</br>'.format(fname))

for grp in ['surface', 'vehicle', 'vehicle_type', 'baby_age']:
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Seat Pan Acceleration Distribution Grouped By: {}'.format(grp))
    ax = stats_df.groupby(grp).boxplot(column='SeatBot_acc_ver_rms',
                                       subplots=False, rot=45, ax=ax)
    fname = 'SeatBot_acc_ver-by-{}-boxplot.png'.format(grp)
    fig.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
    boxp_html.append('<img src="fig/{}"></img>'.format(fname))

html_source = html_tmpl.format(
    boxp_html='\n  '.join(boxp_html),
    mean_table=mean_df.to_frame().to_html(),
    sess_html='\n  '.join(sess_html),
    spect_html='\n  '.join(spect_html),
    trial_html='\n  '.join(trial_html),
    trial_table=stats_df.to_html(),
)
with open(os.path.join(PATH_TO_REPO, 'index.html'), 'w') as f:
    f.write(html_source)

plt.close('all')
