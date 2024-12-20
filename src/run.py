import os
import subprocess

from paths import PATH_TO_REPO

SIGNAL = 'SeatBotacc_ver'  # script currently only processes a single signal
SAMPLE_RATE = 400  # down sample data to this rate

path_to_process = os.path.join(PATH_TO_REPO, 'src', 'process.py')
path_to_website = os.path.join(PATH_TO_REPO, 'src', 'website.py')

if __name__ == '__main__':
    # run each chunk of sessions in a seperate process that must exit so all
    # memory is freed, running all sessions in the same process has a memory
    # leak
    for start_num, end_num in ((0, 5), (5, 10), (10, 15), (15, 99)):
        subprocess.call(['python', path_to_process, str(start_num),
                         str(end_num), SIGNAL, str(SAMPLE_RATE)])

    subprocess.call(['python', path_to_website])
