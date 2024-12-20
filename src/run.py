import os
import subprocess

from paths import PATH_TO_REPO

SIGNAL = 'SeatBotacc_ver'  # script currently only processes a single signal
SAMPLE_RATE = 400  # down sample data to this rate

for start_num, end_num in ((0, 5), (5, 10), (10, 15), (15, 22)):
    path_to_process = os.path.join(PATH_TO_REPO, 'src', 'process.py')
    subprocess.call(['python', path_to_process, str(start_num), str(end_num),
                     SIGNAL, str(SAMPLE_RATE)])
