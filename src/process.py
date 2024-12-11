import os

import yaml

from data import Session

PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

with open(os.path.join(PATH_TO_REPO, 'data', 'sessions.yml')) as f:
    session_meta_data = yaml.safe_load(f)

session_labels = list(session_meta_data.keys())

for session_label in session_labels:
    print('Loading: ', session_label)
    s = Session(session_label)
    s.rotate_imu_data()
    print(s.trial_bounds)
    del s
