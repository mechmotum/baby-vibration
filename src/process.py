import os
from collections import defaultdict

import yaml
import pandas as pd

from data import Session

PATH_TO_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

with open(os.path.join(PATH_TO_REPO, 'data', 'sessions.yml')) as f:
    session_meta_data = yaml.safe_load(f)

session_labels = list(session_meta_data.keys())

motion_trials = ['stoeptegels', 'klinkers', 'tarmac', 'Aula', 'pave']

stats_data = defaultdict(list)

for session_label in session_labels:
    print('Loading: ', session_label)
    s = Session(session_label)
    s.rotate_imu_data()
    print(s.trial_bounds)
    s.calculate_travel_speed()
    for mot_trial in motion_trials:
        if mot_trial in s.trial_bounds:
            for trial_num in s.trial_bounds[mot_trial]:
                trial_df = s.extract_trial(mot_trial, trial_number=trial_num)
                stats_data['surface'].append(mot_trial)
                stats_data['vehicle'].append(s.meta_data['vehicle'])
                stats_data['vehicle_type'].append(s.meta_data['vehicle_type'])
                stats_data['baby_age'].append(s.meta_data['baby_age'])
                stats_data['speed_avg'].append(trial_df['Speed'].mean())
                stats_data['speed_std'].append(trial_df['Speed'].std())
                del trial_df  # critical as this seems to be a copy!
    del s

print(pd.DataFrame(stats_data))
