import os
import pickle
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paths import PATH_TO_REPO, PATH_TO_DATA_DIR, PATH_TO_FIG_DIR
from html_templates import INDEX, H2, P, IMG
from run import SIGNAL

SIGNAL_RMS = SIGNAL + '_rms'
KPH2MPS, MPS2KPH = 1.0/3.6, 3.6
MM2INCH = 1.0/25.4
MAXWIDTH = 160  # mm (max width suitable for an A4 with 25 mm margins)

sns.set_theme(style='white')

with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'rb') as f:
    html_data = pickle.load(f)
with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'rb') as f:
    stats_data = pickle.load(f)

stats_df = pd.DataFrame(stats_data)
# NOTE : it is possible to add duplicate rows when running process.py, so drop
# duplicates
stats_df.drop_duplicates(inplace=True)

stats_df['duration_weight'] = (stats_df['Duration [s]'] /
                               stats_df['Duration [s]'].max())

# target speeds were 5, 12, 20, 25 so group as < 8.5, 8.5-16, 16-22.5, >22.5
stats_df['Target Speed [km/h]'] = [0]*len(stats_df)
crit = stats_df['Mean Speed [m/s]'] <= 8.5*KPH2MPS
stats_df.loc[crit, 'Target Speed [km/h]'] = 5
crit = ((stats_df['Mean Speed [m/s]'] > 8.5*KPH2MPS) &
        (stats_df['Mean Speed [m/s]'] <= 16.0*KPH2MPS))
stats_df.loc[crit, 'Target Speed [km/h]'] = 12
crit = ((stats_df['Mean Speed [m/s]'] > 16.0*KPH2MPS) &
        (stats_df['Mean Speed [m/s]'] <= 22.5*KPH2MPS))
stats_df.loc[crit, 'Target Speed [km/h]'] = 20
crit = stats_df['Mean Speed [m/s]'] > 22.5*KPH2MPS
stats_df.loc[crit, 'Target Speed [km/h]'] = 25

stats_df['Vehicle, Seat, Baby Age'] = (
    stats_df['Vehicle'] + ', ' +
    stats_df['Seat'] + ', ' +
    stats_df['Baby Age [mo]'].astype(str) + ' mo'
)

stats_df['Seat, Baby'] = (
    stats_df['Seat'] + ', ' +
    stats_df['Baby Age [mo]'].astype(str) + ' mo'
)

stats_df['Mean Speed [km/h]'] = stats_df['Mean Speed [m/s]']*3.6
print(stats_df)

groups = ['Vehicle', 'Seat, Baby', 'Road Surface', 'Target Speed [km/h]']
# weight means by duration
weighted_mean_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(
    lambda x: np.average(x, weights=stats_df.loc[x.index, "duration_weight"]))

summary_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(
    **{'Trial Count': 'size',
       'Mean RMS Acceleration [m/s/s]': 'mean'}
)
summary_df['Mean Duration [s]'] = \
    stats_df.groupby(groups)['Duration [s]'].mean()
summary_df['Mean Peak Frequency [Hz]'] = \
    stats_df.groupby(groups)['Peak Frequency [Hz]'].mean()
summary_df['Mean Threshold Frequency [Hz]'] = \
    stats_df.groupby(groups)['Threshold Frequency [Hz]'].mean()
print(summary_df)
print(summary_df.to_latex(float_format="%0.1f"))

# Table that shows how many trials and the mean duration
groups = ['Vehicle Type', 'Road Surface', 'Target Speed [km/h]']
trial_count_df = stats_df.groupby(groups)['Duration [s]'].agg(
    **{'Count': 'count',
       'Mean': 'mean',
       'STD': 'std'})
print(trial_count_df)
print(trial_count_df.to_latex(float_format="%0.1f"))

boxp_html = []

# TODO : This plot also looks pretty good if you swap the x and hue values.
# Maybe better.
boxp_html.append(H2.format('Overall Comparison'))
msg = """Scatter plot of the RMS acceleration of all trials broken down by
vehicle setup (brand, seat configuration, baby age), trial duration, road
surface, and plotted versus speed."""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=stats_df,
    x="Mean Speed [km/h]",
    y="SeatBotacc_ver_rms",
    hue="Vehicle, Seat, Baby Age",
    style='Road Surface',
    size='Duration [s]',
    sizes=(40, 140),
)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Overall VDV Comparison'))
msg = """"""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=stats_df,
    hue="Mean Speed [km/h]",
    y="SeatBotacc_ver_vdv",
    x="Vehicle, Seat, Baby Age",
    style='Road Surface',
    size='Duration [s]',
    sizes=(40, 140),
)
p.set_xticklabels(sorted(stats_df["Vehicle, Seat, Baby Age"].unique()),
                  rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration VDV [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-vdv-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Peak Frequency'))
msg = """TODO"""
boxp_html.append(P.format(msg))
p = sns.boxplot(
    data=stats_df,
    x="Target Speed [km/h]",
    y="Peak Frequency [Hz]",
    hue="Road Surface",
)
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*2/3*MM2INCH))
p.figure.set_layout_engine('constrained')
p.axvline(0.5, color='k')
p.axvline(1.5, color='k')
p.axvline(2.5, color='k')
fname = '{}-peak-freq-dist.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Bandwidth'))
msg = """TODO"""
boxp_html.append(P.format(msg))
p = sns.boxplot(
    data=stats_df,
    x="Target Speed [km/h]",
    y="Threshold Frequency [Hz]",
)
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH/2*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-thresh-freq-dist.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Cargo Bicycle Speed Comparison'))
msg = """How does vibration vary across speed for the cargo bicycles? This plot
shows a linear regression of vertical acceleration versus speed for both
asphalt and paver bricks. The shaded regions represent the 95% confidence
intervals."""
boxp_html.append(P.format(msg))
p = sns.lmplot(
    data=stats_df[stats_df['Vehicle Type'] == 'bicycle'],
    x="Mean Speed [km/h]",
    y="SeatBotacc_ver_rms",
    hue="Road Surface",
    n_boot=200,  # increase to ensure consistent shaded bounds
)
p.legend.draw_frame(True)
p.legend.get_frame().set_linewidth(1.0)
p.legend.get_frame().set_edgecolor('k')
p.set_ylabels(r'Vertical Acceleration RMS [m/s$^2$]')
fname = '{}-bicycle-speed-compare.png'.format(SIGNAL)
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH/2*MM2INCH))
p.figure.set_layout_engine('constrained')
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Vehicle Comparison'))
msg = """Do vehicles differ in each surface-speed scenario?"""
boxp_html.append(P.format(msg))
p = sns.catplot(
    data=stats_df,
    x="Road Surface",
    y="SeatBotacc_ver_rms",
    hue="Vehicle",
    col='Target Speed [km/h]',
    col_wrap=2,
    kind='box',
    order=sorted(stats_df['Road Surface'].unique()),
    sharey=False,
)
p.set_xticklabels(sorted(stats_df['Road Surface'].unique()), rotation=30)
p.figure.set_layout_engine('constrained')
fname = '{}-vehicle-compare.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Stroller Comparison'))
msg = """Compare baby age for each stroller on each tested surface."""
boxp_html.append(P.format(msg))
p = sns.catplot(
    data=stats_df[stats_df['Vehicle Type'] == 'stroller'],
    x="Vehicle",
    y="SeatBotacc_ver_rms",
    hue="Baby Age [mo]",
    col='Road Surface',
    col_wrap=2,
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

boxp_html.append(H2.format('Cargo Bicycle Seat Comparison'))
msg = """Compare the baby seats used in the cargo bicycles for each bicycle and
road surface type."""
boxp_html.append(P.format(msg))
p = sns.catplot(
    data=stats_df[stats_df['Vehicle Type'] == 'bicycle'],
    x="Vehicle",
    y="SeatBotacc_ver_rms",
    hue="Seat, Baby",
    col='Road Surface',
    kind='strip',
    palette='deep',
    sharex=False,
    size=10,
    linewidth=1,
    marker="D",
)
fname = '{}-bicycle-compare.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Baby Mass Comparison'))
msg = 'Compare accelerations across baby mass (age) and speed.'
boxp_html.append(P.format(msg))
p = sns.catplot(
    data=stats_df,
    x="Baby Age [mo]",
    y="SeatBotacc_ver_rms",
    hue='Mean Speed [km/h]',
    col='Vehicle Type',
    sharex=False,
)
p.set_ylabels(r'Vertical Acceleration RMS [m/s$^2$]')
fname = '{}-baby-mass-compare.png'.format(SIGNAL)
# TODO : The legend overlaps the axes if I try to make it a fixed width.
#p.figure.set_size_inches((MAXWIDTH*MM2INCH, 80*MM2INCH))
#p.figure.set_layout_engine('constrained')
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Road Surface Comparison'))
msg = ''
boxp_html.append(P.format(msg))
p = sns.stripplot(
    data=stats_df,
    x="Road Surface",
    y="SeatBotacc_ver_rms",
    hue='Mean Speed [km/h]',
    order=sorted(stats_df['Road Surface'].unique()),
)
p.set_xticklabels(p.get_xticklabels(), rotation=30)
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
fname = '{}-road-surface-compare.png'.format(SIGNAL)
p.figure.set_size_inches((MAXWIDTH*MM2INCH, 100*MM2INCH))
p.figure.set_layout_engine('constrained')
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Stroller Type Comparison'))
msg = ''
boxp_html.append(P.format(msg))
p = sns.pointplot(
    data=stats_df[stats_df['Vehicle Type'] == 'stroller'],
    x='Road Surface',
    y="SeatBotacc_ver_rms",
    hue='Vehicle',
    dodge=True,
    order=sorted(stats_df['Road Surface'].unique()),
)
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.set_xticklabels(p.get_xticklabels(), rotation=30)
fname = '{}-stroller-type-compare.png'.format(SIGNAL)
p.figure.set_size_inches((MAXWIDTH*MM2INCH, 100*MM2INCH))
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Bicycle Comparison'))
msg = ''
boxp_html.append(P.format(msg))
p = sns.lmplot(
    data=stats_df[stats_df['Vehicle Type'] == 'bicycle'],
    x='Mean Speed [km/h]',
    y="SeatBotacc_ver_rms",
    hue="Vehicle",
    col='Road Surface',
    x_bins=3,
    seed=924,
    facet_kws={'sharey': False},
)
p.set_ylabels(r'Vertical Acceleration RMS [m/s$^2$]')
fname = '{}-bicycle-type-compare.png'.format(SIGNAL)
#p.figure.set_size_inches((MAXWIDTH*MM2INCH, 100*MM2INCH))
#p.figure.set_layout_engine('constrained')
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

plt.close('all')

html_source = INDEX.format(
    date=datetime.datetime.today(),
    signal=SIGNAL,
    boxp_html='\n  '.join(boxp_html),
    mean_table=summary_df.to_html(float_format="%0.2f"),
    sess_html='\n  '.join(html_data['sess_html']),
    spec_html='\n  '.join(html_data['spec_html']),
    trial_html='\n  '.join(html_data['trial_html']),
    srot_html='\n  '.join(html_data['srot_html']),
    sync_html='\n  '.join(html_data['sync_html']),
    trial_table=stats_df.to_html(),
)
with open(os.path.join(PATH_TO_REPO, 'index.html'), 'w') as f:
    f.write(html_source)
