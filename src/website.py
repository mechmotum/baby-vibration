# builtin
import datetime
import os
import pickle
import pprint
import subprocess

# external dependencies
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sma
import statsmodels.formula.api as smf

# local
from functions import print_header, eval_health
from html_templates import INDEX, H2, H4, P, IMG
from paths import PATH_TO_REPO, PATH_TO_DATA_DIR, PATH_TO_FIG_DIR
from run import SIGNAL

SIGNAL_RMS = SIGNAL + '_rms'
SIGNAL_RMS_ISO = SIGNAL + '_rms_iso'
SIGNAL_RMS_MAG_ISO = SIGNAL.split('_')[0] + '_rms_mag_iso'
KPH2MPS, MPS2KPH = 1.0/3.6, 3.6
MM2INCH = 1.0/25.4
MAXWIDTH = 160.0  # mm (max width suitable for an A4 with 25 mm margins)
COMFORT_BOUNDS = (
    (0.0, 0.315, 'not uncomfortable'),
    (0.315, 0.63, 'a little uncomfortable'),
    (0.5, 1.0, 'fairly uncomfortable'),
    (0.8, 1.6, 'uncomfortable'),
    (1.25, 2.5, 'very uncomfortable'),
    (2.0, 99.0, 'extremely uncomfortable'),
)

sns.set_theme(style='whitegrid')

try:
    githash = subprocess.check_output([
        'git', 'rev-parse', 'HEAD']).decode('ascii').strip()
except (FileNotFoundError, subprocess.CalledProcessError):
    githash = 'none found'

with open(os.path.join(PATH_TO_DATA_DIR, 'html-data.pkl'), 'rb') as f:
    html_data = pickle.load(f)
with open(os.path.join(PATH_TO_DATA_DIR, 'stats-data.pkl'), 'rb') as f:
    stats_data = pickle.load(f)

stats_df = pd.DataFrame(stats_data)
# NOTE : It may be possible to add duplicate rows when running process.py, so
# drop duplicates.
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

bicycle_df = stats_df[stats_df['Vehicle Type'] == 'bicycle']
stroller_df = stats_df[stats_df['Vehicle Type'] == 'stroller']

print_header("Statistics Data Frame in Tidy Format")
print("All columns:")
pprint.pprint(stats_df.columns.to_list())
print(stats_df)

groups = ['Vehicle', 'Seat, Baby', 'Road Surface', 'Target Speed [km/h]']
# weight means by duration
weighted_mean_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(
    lambda x: np.average(x, weights=stats_df.loc[x.index, "duration_weight"]))

summary_df = stats_df.groupby(groups)[SIGNAL_RMS].agg(
    **{'Trial Count': 'size',
       'RMS Acceleration [m/s/s]': 'mean'}
)
summary_df['ISO Weighted RMS Acceleration [m/s/s]'] = \
    stats_df.groupby(groups)[SIGNAL_RMS_ISO].mean()
summary_df['ISO Weighted Peak Frequency [Hz]'] = \
    stats_df.groupby(groups)['Peak Frequency [Hz]'].mean()
summary_df['ISO Weighted Bandwidth (80%) [Hz]'] = \
    stats_df.groupby(groups)['Threshold Frequency [Hz]'].mean()
summary_df['Duration [s]'] = \
    stats_df.groupby(groups)['Duration [s]'].mean()
summary_df['Crest Factor'] = stats_df.groupby(groups)['Crest Factor'].mean()
print_header("Means Per Scenario")
print(summary_df)
print(summary_df.to_latex(float_format="%0.1f"))

# Table that shows how many trials and the mean duration
groups = ['Vehicle Type', 'Road Surface', 'Target Speed [km/h]']
trial_count_df = stats_df.groupby(groups)['Duration [s]'].agg(
    **{'Count': 'count',
       'Mean': 'mean',
       'STD': 'std'})
print_header("Trial Counts and Duration Stats Per Scenario")
print(trial_count_df)
print(trial_count_df.to_latex(float_format="%0.1f"))

f = (f"{SIGNAL_RMS_ISO} ~ "
     "Q('Mean Speed [m/s]') * "
     "C(Q('Road Surface'), Treatment('Tarmac')) + "
     "C(Q('Vehicle, Seat, Baby Age'), Treatment('trike, maxicosi, 3 mo'))")
mod = smf.ols(formula=f, data=bicycle_df)
bicycle_res = mod.fit()
print_header("Bicycle OLS Results (Vertical ISO Weigthed RMS)")
print(bicycle_res.summary())
bicycle_comp_tables = []
for surf in ('Tarmac', 'Paver Bricks'):
    for speed in ('Low', 'High'):
        print_header(f"Pairwise Comparison of Vehicle Setups on {surf} "
                     f"at {speed} Speed")
        if speed == 'Low':
            spd_sel = bicycle_df['Target Speed [km/h]'] < 15
        if speed == 'High':
            spd_sel = bicycle_df['Target Speed [km/h]'] > 15
        sub_df = bicycle_df[(bicycle_df['Road Surface'] == surf) & spd_sel]
        comp_tab = pairwise_tukeyhsd(sub_df[f"{SIGNAL_RMS_ISO}"],
                                     sub_df['Vehicle, Seat, Baby Age'])
        bicycle_comp_tables.append((surf, speed, comp_tab))
        print(comp_tab)

f = (f"{SIGNAL_RMS_ISO} ~ "
     "C(Q('Road Surface'), Treatment('Tarmac')) + "
     "C(Q('Vehicle, Seat, Baby Age'), Treatment('greenmachine, cot, 0 mo'))")
mod = smf.ols(formula=f, data=stroller_df)
stroller_res = mod.fit()
print_header("Stroller OLS Results (Vertical ISO Weigthed RMS)")
print(stroller_res.summary())
anova = sma.stats.anova_lm(stroller_res)
print(anova)
stroller_comp_tables = []
for surf in stroller_df['Road Surface'].unique():
    print_header(f"Pairwise Comparison of Vehicle Setups on {surf}")
    sf_sel = stroller_df[stroller_df['Road Surface'] == surf]
    comp_tab = pairwise_tukeyhsd(sf_sel[f"{SIGNAL_RMS_ISO}"],
                                 sf_sel['Vehicle, Seat, Baby Age'])
    stroller_comp_tables.append((surf, comp_tab))
    print(comp_tab)

boxp_html = []

#########################################################
# Figure: Health ISO Weighted RMS All Trials Scatter Plot
#########################################################
boxp_html.append(H2.format('All Trials ISO Weighted RMS Compared (Health)'))
msg = f"""Scatter plot of the ISO 2631-1 Weighted RMS {SIGNAL} of all trials
broken down by vehicle setup (brand, seat configuration, baby age), trial
duration, road surface, and plotted versus speed. Black horizontal line
indicate "above the health caution zone" from the ISO 2631-1 standard for
different durations of daily dosage."""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=stats_df.sort_values(["Vehicle, Seat, Baby Age", "Road Surface"]),
    x="Mean Speed [km/h]",
    y=SIGNAL_RMS_ISO,
    style="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    size='Duration [s]',
    sizes=(40, 140),
)
# health risk lines for different durations
for val, note in zip((10.0, 20.0, 60.0, 240.0),
                     ('10 min', '20 min', '1 hr', '4 hr')):
    p.axes.axhline(eval_health(val)[1], color='grey')
    p.axes.text(7.0, eval_health(val)[1] + 0.2, note,
                bbox=dict(boxstyle='round,pad=0.1', color='white'))
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, 1.05*MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

############################################################
# Figure: Health ISO Weighted RMS Stroller Trials Strip Plot
############################################################
boxp_html.append(
    H2.format(f'Stroller ISO Weighted RMS {SIGNAL} Comparison (Health)'))
p = sns.stripplot(
    data=stroller_df,
    y=SIGNAL_RMS_ISO,
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    hue_order=sorted(stroller_df["Road Surface"].unique()),
    order=sorted(stroller_df["Vehicle, Seat, Baby Age"].unique()),
)
# health risk lines for different durations Equation B.1
for val, note in zip((10.0, 20.0, 60.0, 240.0),
                     ('10 min', '20 min', '1 hr', '4 hr')):
    p.axes.axhline(eval_health(val)[1], color='grey')
    p.axes.text(1.15, eval_health(val)[1] + 0.05, note)
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-rms-stroller-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

###################################################################
# Figure: Health ISO Weighted RMS Cargo Bicycle Trials Scatter Plot
###################################################################
boxp_html.append(
    H2.format(f'Cargo Bicycle ISO Weighted RMS {SIGNAL} Comparison (Health)'))
msg = """"""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=bicycle_df.sort_values(["Vehicle, Seat, Baby Age", "Road Surface"]),
    y=SIGNAL_RMS_ISO,
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    palette=['C1', 'C4', 'C0', 'C2', 'C3', 'C5'],
    style="Target Speed [km/h]",
)
# health risk lines for different durations
for val, note in zip((10.0, 20.0, 60.0, 240.0),
                     ('10 min', '20 min', '1 hr', '4 hr')):
    p.axes.axhline(eval_health(val)[1], color='grey')
    p.axes.text(1.1, eval_health(val)[1] + 0.15, note,
                bbox=dict(boxstyle='round,pad=0.02', color='white'))
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-rms-bicycle-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

##########################################################
# Figure: Comfort ISO Weighted RMS All Trials Scatter Plot
##########################################################
boxp_html.append(H2.format('All Trials ISO Weighted RMS Compared (Comfort)'))
msg = f"""Scatter plot of the ISO 2631-1 Weighted RMS {SIGNAL.split('_')[0]}
magnitude of all trials broken down by vehicle setup (brand, seat
configuration, baby age), trial duration, road surface, and plotted versus
speed. Horizontal lines indicate the ISO 2631-1 comfort labels."""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=stats_df.sort_values(["Vehicle, Seat, Baby Age", "Road Surface"]),
    x="Mean Speed [km/h]",
    y=SIGNAL_RMS_MAG_ISO,
    style="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    size='Duration [s]',
    sizes=(40, 140),
)
# comfort lines for different durations
for low, high, note in COMFORT_BOUNDS:
    p.axes.axhline(low, color='grey')
    p.axes.text(7.0, low + 0.05, note, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.0', color='white'))
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Acceleration Magnitude RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, 1.4*MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-compare-all-comfort.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()

#############################################################
# Figure: Comfort ISO Weighted RMS Stroller Trials Strip Plot
#############################################################
boxp_html.append(IMG.format('', fname) + '\n</br>')
boxp_html.append(H2.format(f'Stroller ISO Weighted RMS {SIGNAL} Comparison'))
p = sns.stripplot(
    data=stroller_df,
    y=SIGNAL_RMS_MAG_ISO,
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    hue_order=sorted(stroller_df["Road Surface"].unique()),
    order=sorted(stroller_df["Vehicle, Seat, Baby Age"].unique()),
)
for low, high, note in COMFORT_BOUNDS:
    p.axes.axhline(low, color='black')
    p.axes.text(1.3, low + 0.05, note, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.0', color='white'))
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, 1.1*MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-rms-comfort-stroller-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

###################################################################
# Figure: Health ISO Weighted RMS Cargo Bicycle Trials Scatter Plot
###################################################################
boxp_html.append(
    H2.format(f'Cargo Bicycle ISO Weighted RMS {SIGNAL} Comparison'))
msg = """"""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=bicycle_df.sort_values("Vehicle, Seat, Baby Age"),
    y=SIGNAL_RMS_MAG_ISO,
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    style="Target Speed [km/h]",
)
for low, high, note in COMFORT_BOUNDS:
    p.axes.axhline(low, color='black')
    p.axes.text(1.4, low + 0.1, note, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.0', color='white'))
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration RMS [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, 1.2*MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-rms-comfort-bicycle-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

###################################
boxp_html.append(H2.format(f'All Trials Unweighted {SIGNAL} VDV Compared'))
msg = """"""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=stats_df.sort_values("Vehicle, Seat, Baby Age"),
    y=SIGNAL + "_vdv",
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    size="Mean Speed [km/h]",
)
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration VDV [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-vdv-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format(f'Stroller Unweighted {SIGNAL} VDV Comparison'))
msg = """"""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=stroller_df.sort_values("Vehicle, Seat, Baby Age"),
    y=SIGNAL + "_vdv",
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    size="Mean Speed [km/h]",
)
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration VDV [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-vdv-stroller-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(
    H2.format(f'Cargo Bicycle Unweighted {SIGNAL} VDV Comparison'))
msg = """"""
boxp_html.append(P.format(msg))
p = sns.scatterplot(
    data=bicycle_df.sort_values("Vehicle, Seat, Baby Age"),
    y=SIGNAL + "_vdv",
    x="Vehicle, Seat, Baby Age",
    hue='Road Surface',
    size="Mean Speed [km/h]",
)
p.set_xticklabels(p.get_xticklabels(), rotation=90)
sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
p.set_ylabel(r'Vertical Acceleration VDV [m/s$^2$]')
p.figure.set_size_inches((MAXWIDTH*MM2INCH, MAXWIDTH*MM2INCH))
p.figure.set_layout_engine('constrained')
fname = '{}-vdv-bicycle-compare-all.png'.format(SIGNAL)
p.figure.savefig(os.path.join(PATH_TO_FIG_DIR, fname))
plt.clf()
boxp_html.append(IMG.format('', fname) + '\n</br>')

boxp_html.append(H2.format('Peak Frequency'))
msg = """"""
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

boxp_html.append(H2.format('Bandwidth (80%)'))
msg = """"""
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
msg = f"""How does vibration vary across speed for the cargo bicycles? This plot
shows a linear regression of ISO weighted {SIGNAL} versus speed for both
asphalt and paver bricks. The shaded regions represent the 95% confidence
intervals."""
boxp_html.append(P.format(msg))
p = sns.lmplot(
    data=bicycle_df,
    x="Mean Speed [km/h]",
    y=SIGNAL_RMS_ISO,
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
    y=SIGNAL_RMS_ISO,
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
    data=stroller_df,
    x="Vehicle",
    y=SIGNAL_RMS_ISO,
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
    data=bicycle_df,
    x="Vehicle",
    y=SIGNAL_RMS_ISO,
    hue="Seat, Baby",
    col='Road Surface',
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

boxp_html.append(H2.format('Baby Mass Comparison'))
msg = 'Compare accelerations across baby mass (age) and speed.'
boxp_html.append(P.format(msg))
p = sns.catplot(
    data=stats_df,
    x="Baby Age [mo]",
    y=SIGNAL_RMS_ISO,
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
    y=SIGNAL_RMS_ISO,
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
    data=stroller_df,
    x='Road Surface',
    y=SIGNAL_RMS_ISO,
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
    data=bicycle_df,
    x='Mean Speed [km/h]',
    y=SIGNAL_RMS_ISO,
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
    githash=githash,
    signal=SIGNAL,
    boxp_html='\n  '.join(boxp_html),
    bicycle_stats=bicycle_res.summary().as_html(),
    stroller_stats=stroller_res.summary().as_html(),
    stroller_comp='\n'.join([H4.format(surf) + '\n' + tab.summary().as_html() for surf, tab in stroller_comp_tables]),
    bicycle_comp='\n'.join([H4.format(surf + ' ' + speed) + '\n' + tab.summary().as_html() for surf, speed, tab in bicycle_comp_tables]),
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
