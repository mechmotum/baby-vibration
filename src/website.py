import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paths import PATH_TO_REPO, PATH_TO_DATA_DIR, PATH_TO_FIG_DIR
from html_templates import INDEX, H2, H3, HR, P, IMG
from run import SIGNAL

SIGNAL_RMS = SIGNAL + '_rms'

with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'rb') as f:
    html_data = pickle.load(f)
with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'rb') as f:
    stats_data = pickle.load(f)

stats_df = pd.DataFrame(stats_data)
stats_df.drop_duplicates(inplace=True)  # in case I ran things double

stats_df['duration_weight'] = stats_df['duration']/stats_df['duration'].max()

# target speeds were 5, 12, 20, 25
# so group < 8.5, 8.5-16, 16-22.5, >22.5
# 8.5 km/hr = 2.4 m/s
# 16 km/hr = 4.4 m/s
# 22.5 km/hr = 6.3 m/s
stats_df['target_speed'] = [0]*len(stats_df)
stats_df.loc[stats_df['speed_avg'] <= 2.4, 'target_speed'] = 5
stats_df.loc[(stats_df['speed_avg'] > 2.4) &
             (stats_df['speed_avg'] <= 4.4), 'target_speed'] = 12
stats_df.loc[(stats_df['speed_avg'] > 4.4) &
             (stats_df['speed_avg'] <= 6.3), 'target_speed'] = 20
stats_df.loc[stats_df['speed_avg'] > 6.3, 'target_speed'] = 25

stats_df['vehicle_baby'] = (stats_df['vehicle'] + '_' +
                            stats_df['seat'] + '_' +
                            stats_df['baby_age'].astype(str) + 'm')

stats_df['speed_avg_kph'] = stats_df['speed_avg']*3.6

print(stats_df)
groups = ['vehicle', 'baby_age', 'surface', 'target_speed']
# weight means by duration
mean_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(
    lambda x: np.average(x, weights=stats_df.loc[x.index, "duration_weight"]))
print(mean_df)
summary_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(
    count='size',
    weighted_mean=lambda x: np.average(
        x, weights=stats_df.loc[x.index, "duration_weight"]),
    std='std')  # TODO : weighted std https://stackoverflow.com/a/2415343
print(summary_df)

boxp_html = []

boxp_html.append(H2.format('Speed Comparison'))
boxp_html.append(P.format('How does vibration vary across speed?'))
p = sns.scatterplot(
    data=stats_df,
    x="speed_avg_kph",
    y="SeatBotacc_ver_rms",
    hue="vehicle_baby",
    style='surface',
)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
plt.gcf().set_size_inches((12, 8))
plt.tight_layout()
fname = '{}-speed-compare.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Vehicle Comparison'))
boxp_html.append(P.format('Do vehicles-baby combinations differ in each '
                          'surface-speed scenario?'))
p = sns.catplot(
    data=stats_df,
    x="surface",
    y="SeatBotacc_ver_rms",
    hue="vehicle",
    col='target_speed',
    col_wrap=2,
    kind='box',
    sharex=False,
    sharey=False,
)
fname = '{}-vehicle-compare.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Stroller Comparison'))
boxp_html.append(P.format('Do strollers differ in each '
                          'surface scenario?'))
p = sns.catplot(
    data=stats_df[stats_df['vehicle_type'] == 'stroller'],
    x="vehicle",
    y="SeatBotacc_ver_rms",
    hue="baby_age",
    col='surface',
    col_wrap=3,
    kind='strip',
    palette='deep',
    sharex=False,
    sharey=False,
    size=10,
    linewidth=1,
    marker="D",
)
fname = '{}-stroller-compare.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Bike/Trike Comparison'))
boxp_html.append(P.format('Do cargo bikes differ in each '
                          'surface scenario?'))
p = sns.catplot(
    data=stats_df[stats_df['vehicle_type'] == 'bicycle'],
    x="vehicle",
    y="SeatBotacc_ver_rms",
    hue="baby_age",
    col='surface',
    kind='strip',
    palette='deep',
    sharex=False,
    sharey=False,
    size=10,
    linewidth=1,
    marker="D",
)
fname = '{}-bicycle-compare.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

plt.close('all')

html_source = INDEX.format(
    signal=SIGNAL,
    boxp_html='\n  '.join(boxp_html),
    mean_table=mean_df.to_frame().to_html(),
    sess_html='\n  '.join(html_data['sess_html']),
    spec_html='\n  '.join(html_data['spec_html']),
    trial_html='\n  '.join(html_data['trial_html']),
    srot_html='\n  '.join(html_data['srot_html']),
    sync_html='\n  '.join(html_data['sync_html']),
    trial_table=stats_df.to_html(),
)
with open(os.path.join(PATH_TO_REPO, 'index.html'), 'w') as f:
    f.write(html_source)
