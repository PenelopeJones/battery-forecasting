import sys
sys.path.append('../')
import time
import pickle

from utils.exp_util import extract_data, extract_input
from utils.models import XGBModel

import pdb

channels = [1, 2, 3, 4, 5, 6, 7, 8]
params = {'max_depth':10,
          'n_splits':2,
          'n_estimators':25,
          'n_ensembles':2}
experiment = 'variable-discharge'
log_name = 'results/{}/log-next-cycle.txt'.format(experiment)

# Test model using different state representations
input_names = ['actions', 'eis', 'eis-actions', 'cvfs-actions', 'c-actions', 'ct-actions',
               'soh-actions', 'eis-cvfs-ct-c-actions', 'eis-cvfs-ct-actions']

# Extract variable discharge data set
cell_var, cap_ds_var, data_var = extract_data(experiment, channels)

#data = (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates)
print('Number of datapoints = {}'.format(data_var[0].shape[0]))

# output = next cycle discharge capacity
y = cap_ds_var

for i in range(len(input_names)):
    input_name = input_names[i]
    experiment_name = '{}_n1_xgb'.format(input_name)
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(input_name, params['max_depth'], params['n_estimators'],
                                                                                                                                        params['n_ensembles'], params['n_splits'])
    print(experiment_info)
    t0 = time.time()

    # extract relevant inputs
    x = extract_input(input_name, data_var)

    regressor = XGBModel(x, y, cell_var, experiment, experiment_name, n_ensembles=params['n_ensembles'],
                         n_splits=params['n_splits'], max_depth=params['max_depth'],
                         n_estimators=params['n_estimators'])
    regressor.analysis(log_name, experiment_info)
    print('Time taken = {:.2f}'.format(time.time() - t0))

print('Done.')
