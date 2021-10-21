import sys
sys.path.append('../')
import time
import pickle

import numpy as np

from utils.exp_util import extract_data, extract_input, ensemble_predict

import pdb

channels = [1, 2, 3, 4, 5, 6, 7, 8]

# Testing on fixed discharge data set
cell_fixed, cap_ds_fixed, data_fixed = extract_data('fixed-discharge', channels)
x_fixed = extract_input(input_name, data_fixed)
pred_fixed, pred_fixed_err = ensemble_predict(x_fixed, 'variable-discharge', input_name)


dts = '../results/fixed-discharge'
np.save('{}/predictions/pred_mn_{}.npy'.format(dts, input_name), pred_fixed)
np.save('{}/predictions/pred_std_{}.npy'.format(dts, input_name), pred_fixed_err)
np.save('{}/predictions/true_{}.npy'.format(dts, input_name), cap_ds_fixed)
