import os

import yaml

with open('config.yml') as f:
    config_data = yaml.safe_load(f)

PATH_TO_SESSION_DATA = config_data['data-directory']
PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
PATH_TO_DATA_DIR = os.path.join(PATH_TO_REPO, 'data')
PATH_TO_FIG_DIR = os.path.join(PATH_TO_REPO, 'fig')
PATH_TO_BOUNDS_DIR = os.path.join(PATH_TO_FIG_DIR, 'bounds')
PATH_TO_TIME_HIST_DIR = os.path.join(PATH_TO_FIG_DIR, 'time_hist')
PATH_TO_SPECT_DIR = os.path.join(PATH_TO_FIG_DIR, 'spectrums')
PATH_TO_ACCROT_DIR = os.path.join(PATH_TO_FIG_DIR, 'accrot')

paths_to_create = [
    PATH_TO_DATA_DIR,
    PATH_TO_FIG_DIR,
    PATH_TO_BOUNDS_DIR,
    PATH_TO_TIME_HIST_DIR,
    PATH_TO_SPECT_DIR,
    PATH_TO_ACCROT_DIR,
]

for dr in paths_to_create:
    if not os.path.exists(dr):
        os.mkdir(dr)
