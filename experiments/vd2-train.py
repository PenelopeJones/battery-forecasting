import sys
sys.path.append('../')
import time
import pickle

from utils.exp_util_new import extract_data_type2, extract_input
from utils.models import XGBModel

import pdb

channels = [1, 2, 3, 4, 5, 6, 7, 8]
params = {'max_depth':100,
          'n_splits':16,
          'n_estimators':500,
          'n_ensembles':10}

experiment = 'variable-discharge-type2'
channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',]

# Test model using different state representations
input_names = ['eis-cvfs-actions', 'eis-actions']

# Extract variable discharge data set
cell_var, cap_ds_var, data_var = extract_data_type2(experiment, channels)

#data = (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates)
print('Number of datapoints = {}'.format(data_var[0].shape[0]))

# output = next cycle discharge capacity
y = cap_ds_var

for i in range(len(input_names)):
    input_name = input_names[i]
    experiment_name = '{}_n1_xgb2'.format(input_name)
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(input_name, params['max_depth'], params['n_estimators'],
                                                                                                                                        params['n_ensembles'], params['n_splits'])
    print(experiment_info)
    t0 = time.time()

    # extract relevant inputs
    x = extract_input(input_name, data_var, suffix='vd2')

    regressor = XGBModel(x, y, cell_var, experiment, experiment_name, n_ensembles=params['n_ensembles'],
                         n_splits=params['n_splits'], max_depth=params['max_depth'],
                         n_estimators=params['n_estimators'])
    regressor.train_no_predict(x, y)
    print('Time taken = {:.2f}'.format(time.time() - t0))

print('Done.')
