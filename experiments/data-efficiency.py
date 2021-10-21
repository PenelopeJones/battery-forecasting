import sys
sys.path.append('../')
import time
import pickle

import numpy as np
from sklearn.metrics import r2_score

from utils.exp_util import extract_data, extract_input
from utils.models import XGBModel

import pdb

channels = [1, 2, 3, 4, 5, 6, 7, 8]
params = {'max_depth':10,
          'n_splits':2,
          'n_estimators':25,
          'n_ensembles':2}
experiment = 'variable-discharge'
log_name = '../results/{}/log-n-cells.txt'.format(experiment)
input_name = 'eis-actions'
n_cells_list = [2, 4, 8, 16, 20]
p = np.random.permutation(24)
# Extract variable discharge data set
cell_var, cap_ds_var, data_var = extract_data(experiment, channels)
cell_ids = np.unique(cell_var)[p]
x = extract_input(input_name, data_var)

# output = next cycle discharge capacity
y = cap_ds_var
n_splits = params['n_splits']

for n_cells in n_cells_list:
    experiment_name = '{}_{}cells_xgb'.format(input_name, n_cells)
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \t{} cells \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(input_name, n_cells, params['max_depth'], params['n_estimators'],
                                                                                                                                        params['n_ensembles'], params['n_splits'])
    t0 = time.time()
    r2s_tr = []
    r2s_te = []
    pes_tr = []
    pes_te = []

    for split in range(n_splits):
        cell_ids_s = cell_ids[0:n_cells+2]
        experiment_info = '\nInput: {} \tOutput: c(discharge)_n+1 \nMax depth: {}\tSplits:{}\n'.format(input_name, params['max_depth'], n_cells)
        print(cell_ids[0])
        print(cell_ids[1])
        cell_test1 = cell_ids[0]
        cell_test2 = cell_ids[1]
        cell_train = cell_ids[2:n_cells+2]
        idx_test1 = np.where(np.isin(cell_var, cell_test1))
        idx_test2 = np.where(np.isin(cell_var, cell_test2))
        idx_train = np.where(np.isin(cell_var, cell_train))
        x_train = x[idx_train]
        print('Number of datapoints = {}'.format(x_train.shape[0]))
        y_train = cap_ds_var[idx_train]
        x_test1 = x[idx_test1]
        y_test1 = cap_ds_var[idx_test1]
        x_test2 = x[idx_test2]
        y_test2 = cap_ds_var[idx_test2]

        regressor = XGBModel(None, None, cell_ids_s, experiment, experiment_name, n_ensembles=params['n_ensembles'],
                             n_splits=params['n_splits'], max_depth=params['max_depth'],
                             n_estimators=params['n_estimators'])
        y_pred_tr, y_pred_tr_err, y_pred_te1, y_pred_te1_err, y_pred_te2, y_pred_te2_err, = regressor.train_and_predict(x_train, y_train, x_test1,
                                                                                                                        cell_test1, x_test2, cell_test2)

        dts = '../results/{}'.format(self.experiment)
        # save test cell predictions
        np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_pred_te1)
        np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_pred_te1_err)
        np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_test1)
        np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_pred_te2)
        np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_pred_te2_err)
        np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_test2)

        r2s_tr.append(r2_score(y_train, y_pred_tr))
        r2s_te.append(r2_score(y_test1, y_pred_te1))
        r2s_te.append(r2_score(y_test2, y_pred_te2))
        pes_tr.append(np.abs(y_train - y_pred_tr) / y_train)
        pes_te.append(np.abs(y_test1 - y_pred_te1) / y_test1)
        pes_te.append(np.abs(y_test2 - y_pred_te2) / y_test2)
        cell_ids = np.roll(cell_ids, 2)

    r2_tr = np.median(np.array(r2s_tr))
    r2_te = np.median(np.array(r2s_te))
    pe_tr = 100*np.median(np.hstack(pes_tr).reshape(-1))
    pe_te = 100*np.median(np.hstack(pes_te).reshape(-1))
    print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))
    with open(log_name, 'a+') as file:
        file.write(experiment_info)
        file.write('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}\n'.format(r2_tr, pe_tr, r2_te, pe_te))
        file.flush()
print('Done.')
