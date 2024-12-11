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

motion_trials = ['stoeptegels']

stats_table_data = defaultdict(list)

for session_label in session_labels:
    print('Loading: ', session_label)
    s = Session(session_label)
    s.rotate_imu_data()
    print(s.trial_bounds)
    s.calculate_travel_speed()
    if 'stoeptegels' in s.trial_bounds:
        for trial_num in s.trial_bounds['stoeptegels']:
            trial_df = s.extract_trial('stoeptegels', trial_number=trial_num)
            stats_table_data['Surface'].append('stoeptegels')
            stats_table_data['Vehicle'].append(s.meta_data['vehicle'])
            stats_table_data['Vehicle Type'].append(s.meta_data['vehicle_type'])
            stats_table_data['Baby Age'].append(s.meta_data['baby_age'])
            stats_table_data['Mean Speed'].append(trial_df['Speed'].mean())
            del trial_df  # critical as this seems to be a copy!
    del s

print(pd.DataFrame(stats_table_data))
