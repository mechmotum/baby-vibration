import os
from collections import defaultdict
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data import Session, plot_frequency_spectrum, datetime2seconds
from data import PATH_TO_REPO, PATH_TO_DATA_DIR, PATH_TO_FIG_DIR

PATH_TO_BOUNDS_DIR = os.path.join(PATH_TO_FIG_DIR, 'bounds')
PATH_TO_TIME_HIST_DIR = os.path.join(PATH_TO_FIG_DIR, 'time_hist')
PATH_TO_SPECT_DIR = os.path.join(PATH_TO_FIG_DIR, 'spectrums')
PATH_TO_ACCROT_DIR = os.path.join(PATH_TO_FIG_DIR, 'accrot')

for dr in [PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR, PATH_TO_TIME_HIST_DIR,
           PATH_TO_SPECT_DIR, PATH_TO_ACCROT_DIR]:
    if not os.path.exists(dr):
        os.mkdir(dr)

NUM_SESSIONS = None  # None for all
SAMPLE_RATE = 400  # down sample data to this rate
SIGNAL = 'SeatBotacc_ver'  # script currently only processes a single signal
SIGNAL_RMS = SIGNAL + '_rms'

html_tmpl= """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Baby Vehicle Vibration Results</title>
  </head>
  <body>

  <h1>Baby Vehicle Vibration Results</h1>
  <hr>
  <p>
    <strong>Warning: These results are preliminary, do not rely on them until a
    supporting paper is published.</strong>
  </p>
  <p>
    This results page examines the signal: <strong>{signal}</strong>.
  </p>

  <h1>Duration Weighted Mean of the RMS of {signal}</h1>
  <hr>
{mean_table}

  <h1>Box Plots of RMS of {signal} </h1>
  <hr>
{boxp_html}

  <h1>Sessions Segmented into Trials</h1>
  <p>This section shows how the sessions are segmented into trials.</p>
{sess_html}

  <h1>ISO 2631-1 Weights</h1>
  <hr>
  <p>
    Plots of the filter weights versus frequency we apply to the data.
  </p>
  <img src='fig/iso-filter-weights-01.png'</img>
  <img src='fig/iso-filter-weights-02.png'</img>

  <h1>Seat Pan Vertical Acceleration Spectrums</h1>
  <hr>
{spect_html}

  <h1>Sensor Rotations</h1>
  <hr>
{srot_html}

  <h1>Trials</h1>
  <hr>
{trial_table}

  <h1>Seat Pan Vertical Acceleration Time Histories</h1>
  <hr>
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
srot_html = []

for sess_count, session_label in enumerate(session_labels[:NUM_SESSIONS]):
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
        srot_html.append('<h2>' + session_label + '</h2>')
        srot_html.append('<img src="fig/accrot/' +
                         session_label + '.png"></img>')
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
        plt.clf()
        plt.close('all')
        del axes, ax, ax1, ax2
        gc.collect()

        trial_html.append('<h2>{}</h2>'.format(session_label))
        spect_html.append('<h2>{}</h2>'.format(session_label))
        for mot_trial in motion_trials:
            if mot_trial in s.trial_bounds:
                trial_html.append('<h3>{}</h3>'.format(mot_trial))
                spect_html.append('<h3>{}</h3>'.format(mot_trial))
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
                    trial_html.append('<img src="fig/time_hist/' + file_name +
                                      '.png"</img>')

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
                    spect_html.append('<img src="fig/spectrums/' + file_name +
                                      '.png"</img>')

                    plt.clf()
                    plt.close('all')
                    del freq, amp, rms, ax
                    gc.collect()
        del s
        gc.collect()

stats_df = pd.DataFrame(stats_data)
stats_df['duration_weight'] = stats_df['duration']/stats_df['duration'].max()
print(stats_df)
groups = ['vehicle', 'baby_age', 'surface']
# weight means by duration
wm = lambda x: np.average(x, weights=stats_df.loc[x.index, "duration_weight"])
mean_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(wm)
#mean_df = stats_df.groupby(groups)[SIGNAL_RMS].mean()
print(mean_df)

boxp_html = []

boxp_html.append('<h2>SeatBot_acc_ver</h2>')

fig, ax = plt.subplots(5, layout='constrained', figsize=(8, 16))
fig.suptitle('Seat Pan RMS Acceleration Distributions')
stats_df.groupby('surface').boxplot(by=['vehicle', 'baby_age'],
                                    column=SIGNAL_RMS, ax=ax)
fname = 'SeatBot_acc_ver-dist-boxplot.png'
fig.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
boxp_html.append('<img src="fig/{}"></img>\n</br>'.format(fname))

for grp in ['surface', 'vehicle', 'vehicle_type', 'baby_age']:
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Seat Pan Acceleration Distribution Grouped By: {}'.format(grp))
    ax = stats_df.groupby(grp).boxplot(column=SIGNAL_RMS,
                                       subplots=False, rot=45, ax=ax)
    fname = 'SeatBot_acc_ver-by-{}-boxplot.png'.format(grp)
    fig.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
    boxp_html.append('<img src="fig/{}"></img>'.format(fname))

boxp_html.append('<h2>Speed</h2>')
fig, ax = plt.subplots(5, layout='constrained', figsize=(8, 16))
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

html_source = html_tmpl.format(
    signal=SIGNAL,
    boxp_html='\n  '.join(boxp_html),
    mean_table=mean_df.to_frame().to_html(),
    sess_html='\n  '.join(sess_html),
    spect_html='\n  '.join(spect_html),
    trial_html='\n  '.join(trial_html),
    srot_html='\n  '.join(srot_html),
    trial_table=stats_df.to_html(),
)
with open(os.path.join(PATH_TO_REPO, 'index.html'), 'w') as f:
    f.write(html_source)

plt.clf()
plt.close('all')
