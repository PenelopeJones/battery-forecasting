import sys
sys.path.append('../')
import time
import pickle

import numpy as np

from utils.exp_util_new import extract_data_type2, extract_input
from utils.exp_util import ensemble_predict

import pdb

channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',]
input_name = 'eis-actions'
# Testing on fixed discharge data set
exp_test = 'vd2-35C'
exp_train = 'variable-discharge'
cells, cap_ds, data = extract_data_type2(exp_test, channels, suffix='vd2')
x = extract_input(input_name, data, suffix='vd2')
pred, pred_err = ensemble_predict(x, exp_train, input_name)

dts = '../results/{}'.format(exp_test)
np.save('{}/predictions/vd1_pred_mn_{}.npy'.format(dts, input_name), pred)
np.save('{}/predictions/vd1_pred_std_{}.npy'.format(dts, input_name), pred_err)
np.save('{}/predictions/vd1_true_{}.npy'.format(dts, input_name), cap_ds)
