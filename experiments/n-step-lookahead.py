import sys
sys.path.append('../')
import time
import pickle

import numpy as np

from utils.exp_util import extract_data, extract_n_step_data, extract_input, identify_cells
from utils.models import XGBModel

import pdb

channels = [1, 2, 3, 4, 5, 6, 7, 8]
params = {'max_depth':100,
          'n_splits':12,
          'n_estimators':500,
          'n_ensembles':10}
experiment = 'variable-discharge'
log_name = '../results/{}/log-n-step-lookahead.txt'.format(experiment)

# Test model using different state representations
input_name = 'eis-actions'

# Extract variable discharge data set
(states, actions, cycles, cap_ds) = extract_n_step_data(experiment, channels)

cell_map = identify_cells(experiment, channels)

n_steps = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40]

for step in n_steps:
    experiment_name = '{}_n{}_xgb'.format(input_name, step)
    experiment_info = '\nInput: {} \tOutput: Q_n+{} \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(input_name, step, params['max_depth'], params['n_estimators'],
                                                                                                                                        params['n_ensembles'], params['n_splits'])
    print(experiment_info)
    t0 = time.time()

    nl =step-1
    nl_states = []
    nl_actions = []
    nl_caps = []
    nl_idx = []

    for channel in channels:
        cells = cell_map[channel]
        for cell in cells:
            cell_states = states[cell]
            cell_actions = actions[cell]
            cell_cap_ds = cap_ds[cell]

            ns = cell_states.shape[0]

            if nl == 0:
                nl_actions.append(cell_actions)
                nl_states.append(cell_states)
                nl_caps.append(cell_cap_ds)

            else:
                for i in range(ns - nl):
                    nl_actions.append(cell_actions[i:i+nl+1, :].reshape(1, -1))
                nl_states.append(cell_states[:-nl, :])
                nl_caps.append(cell_cap_ds[nl:])
            nl_idx.append([cell]*(ns-nl))

    nl_idx = np.array([item for sublist in nl_idx for item in sublist])
    nl_states = np.vstack(nl_states)
    nl_actions = np.vstack(nl_actions)
    nl_caps = np.hstack(nl_caps)

    x = np.concatenate((nl_states, nl_actions), axis=1)
    y = nl_caps

    regressor = XGBModel(x, y, nl_idx, experiment, experiment_name, n_ensembles=params['n_ensembles'],
                         n_splits=params['n_splits'], max_depth=params['max_depth'],
                         n_estimators=params['n_estimators'])
    regressor.analysis(log_name, experiment_info)
    print('Time taken = {:.2f}'.format(time.time() - t0))

print('Done.')
