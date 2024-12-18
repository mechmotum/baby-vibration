import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import PATH_TO_REPO, PATH_TO_DATA_DIR, PATH_TO_FIG_DIR
from run import SIGNAL, SIGNAL_RMS

with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'r') as f:
    html_data = pickle.load(f)
with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'r') as f:
    stats_data = pickle.load(f)

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

plt.clf()
plt.close('all')

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

with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'r') as f:
    html_data = pickle.load(f)

html_source = html_tmpl.format(
    signal=SIGNAL,
    boxp_html='\n  '.join(boxp_html),
    mean_table=mean_df.to_frame().to_html(),
    sess_html='\n  '.join(html_data['sess_html']),
    spect_html='\n  '.join(html_data['spect_html']),
    trial_html='\n  '.join(html_data['trial_html']),
    srot_html='\n  '.join(html_data['srot_html']),
    trial_table=stats_df.to_html(),
)
with open(os.path.join(PATH_TO_REPO, 'index.html'), 'w') as f:
    f.write(html_source)
