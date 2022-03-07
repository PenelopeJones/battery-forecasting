import sys
sys.path.append('../')
import time
import pickle

import numpy as np

from utils.exp_util_new import extract_data_type2, extract_input, ensemble_predict

import pdb

channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',]
input_name = 'eis-actions'
exp_test = 'vd2-35C'
exp_train = 'variable-discharge-type2'
# Testing on fixed discharge data set
cells, cap_ds, data = extract_data_type2(exp_test, channels, suffix='vd2-35C')
x = extract_input(input_name, data=None, suffix='vd2-35C')
experiment_name = '{}_n1_xgb2'.format(input_name)
pred, pred_err = ensemble_predict(x, exp_train, experiment_name)


dts = '../results/{}'.format(exp_test)
np.save('{}/predictions/pred_mn_{}.npy'.format(dts, input_name), pred)
np.save('{}/predictions/pred_std_{}.npy'.format(dts, input_name), pred_err)
np.save('{}/predictions/true_{}.npy'.format(dts, input_name), cap_ds)
