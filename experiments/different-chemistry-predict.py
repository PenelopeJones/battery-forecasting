import sys
sys.path.append('../')
import time
import pickle

import numpy as np

from utils.exp_util import extract_data, extract_input
from utils.exp_util_new import ensemble_predict

import pdb

channels = [1, 2, 3, 4, 5, 6, 7, 8]
input_name = 'eis-actions'
exp_test = 'variable-discharge'
exp_train = 'variable-discharge-type2'
# Testing on fixed discharge data set
cells, cap_ds, data = extract_data(exp_test, channels)
x = extract_input(input_name, data)
experiment_name = '{}_n1_xgb2'.format(input_name)
pred, pred_err = ensemble_predict(x, exp_train, experiment_name)

dts = '../results/{}'.format(exp_test)
np.save('{}/predictions/vd2_pred_mn_{}.npy'.format(dts, input_name), pred)
np.save('{}/predictions/vd2_pred_std_{}.npy'.format(dts, input_name), pred_err)
np.save('{}/predictions/vd2_true_{}.npy'.format(dts, input_name), cap_ds)
