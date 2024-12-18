import os
from data import PATH_TO_DATA_DIR, PATH_TO_FIG_DIR

PATH_TO_BOUNDS_DIR = os.path.join(PATH_TO_FIG_DIR, 'bounds')
PATH_TO_TIME_HIST_DIR = os.path.join(PATH_TO_FIG_DIR, 'time_hist')
PATH_TO_SPECT_DIR = os.path.join(PATH_TO_FIG_DIR, 'spectrums')
PATH_TO_ACCROT_DIR = os.path.join(PATH_TO_FIG_DIR, 'accrot')

for dr in [PATH_TO_FIG_DIR, PATH_TO_BOUNDS_DIR, PATH_TO_TIME_HIST_DIR,
           PATH_TO_SPECT_DIR, PATH_TO_ACCROT_DIR]:
    if not os.path.exists(dr):
        os.mkdir(dr)

SAMPLE_RATE = 400  # down sample data to this rate
SIGNAL = 'SeatBotacc_ver'  # script currently only processes a single signal
SIGNAL_RMS = SIGNAL + '_rms'
NUM_SESSIONS = None  # None for all
START_SESSION, END_SESSION = 0, 5
#START_SESSION, END_SESSION = 5, 10
#START_SESSION, END_SESSION = 10, 15
#START_SESSION, END_SESSION = 15, 20
#START_SESSION, END_SESSION = 15, 22
