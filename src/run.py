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
    for i in range(15):
        start_num, end_num = i*2, i*2 + 2
        if i == 14:
            end_num = 99
        subprocess.call(['python', path_to_process, str(start_num),
                         str(end_num), SIGNAL, str(SAMPLE_RATE)])

    subprocess.call(['python', path_to_website])
