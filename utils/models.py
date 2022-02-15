# First look at and plot the EIS spectra of the cells over the course of their cycle life to now
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append('../')
import pickle

"""


from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares


import re
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from scipy.integrate import simps
from scipy.stats import iqr
from collections import deque
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import iqr
import copy
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gridspec

from d3pg.battery.util import path_to_file, find_cells, find_cycle, search_files, reverse_action
from d3pg.battery.features import reward_function, cv_features, form_state, normalise_state
import pdb
"""

class XGBModel:
    def __init__(self, X, y, cell_nos, experiment, experiment_name, n_ensembles=10, n_splits=5, start_seed=1, max_depth=25, n_estimators=100):
        np.random.seed(start_seed+10)
        self.start_seed = start_seed
        self.n_ensembles = n_ensembles # number of models to train per train/test split
        self.max_depth = max_depth # maximum depth of XGB model
        self.n_estimators = n_estimators # number of estimators per XGB model
        self.n_splits = n_splits # number of train/test splits
        self.n_split = 0 # initial train/test split
        self.X = X # total dataset (n, in_dim): inputs
        self.y = y # total dataset (n, 1): outputs
        self.cell_nos = cell_nos # total dataset (n, 1): specifies the cell corresponding to each datapoint in (X, y)
        self.cell_idx = np.unique(self.cell_nos) # identify name of cells in datasets
        self.experiment = experiment # name of experiment (e.g. variable-discharge)
        self.experiment_name = experiment_name

        if self.experiment == 'both':
            cell_idx_new = []
            for i in range(len(self.cell_idx)):
                if experiment_map[self.cell_idx[i]] == 'fixed-discharge':
                    cell_idx_new.append(self.cell_idx[i])
            cell_idx_new = np.array(cell_idx_new)
            self.cell_idx = cell_idx_new
        elif self.experiment == 'both-variable':
            cell_idx_new = []
            for i in range(len(self.cell_idx)):
                if experiment_map[self.cell_idx[i]] == 'variable-discharge':
                    cell_idx_new.append(self.cell_idx[i])
            cell_idx_new = np.array(cell_idx_new)
            self.cell_idx = cell_idx_new
        self.cell_idx.sort()
        print(self.cell_idx)

        """
        self.n = X.shape[0] # number of datapoints
        self.n_te = self.n // n_splits + 1 # number of test points
        self.idx = np.random.permutation(self.n)
        """
    def split_into_four(self):

        # leave 4 cells out to test
        cell_test1 = self.cell_idx[self.n_split*4]
        cell_test2 = self.cell_idx[self.n_split*4 + 1]
        cell_test3 = self.cell_idx[self.n_split*4 + 2]
        cell_test4 = self.cell_idx[self.n_split*4 + 3]
        print('Split {}: Test cells {} and {} and {} and {}'.format(self.n_split, cell_test1, cell_test2, cell_test3, cell_test4))

        # identify test cell datapoints
        idx_test1 = np.array(np.where(self.cell_nos == cell_test1)).reshape(-1)
        idx_test2 = np.array(np.where(self.cell_nos == cell_test2)).reshape(-1)
        idx_test3 = np.array(np.where(self.cell_nos == cell_test3)).reshape(-1)
        idx_test4 = np.array(np.where(self.cell_nos == cell_test4)).reshape(-1)
        idx_test = np.hstack([idx_test1, idx_test2, idx_test3, idx_test4]).reshape(-1)

        # identify training cell datapoints
        idx_train = np.delete(np.arange(self.X.shape[0]), idx_test)

        # return train and test datasets
        X_test1 = self.X[idx_test1, :]
        y_test1 = self.y[idx_test1]
        X_test2 = self.X[idx_test2, :]
        y_test2 = self.y[idx_test2]
        X_test3 = self.X[idx_test3, :]
        y_test3 = self.y[idx_test3]
        X_test4 = self.X[idx_test4, :]
        y_test4 = self.y[idx_test4]
        X_train = self.X[idx_train, :]
        y_train = self.y[idx_train]

        return X_train, y_train, X_test1, y_test1, X_test2, y_test2, X_test3, y_test3, X_test4, y_test4, cell_test1, cell_test2, cell_test3, cell_test4

    def split_by_cell(self):

        # leave 2 cells out to test
        cell_test1 = self.cell_idx[self.n_split]
        cell_test2 = self.cell_idx[(int(self.n_splits + self.n_split)%len(self.cell_idx))]
        print('Split {}: Test cells {} and {}'.format(self.n_split, cell_test1, cell_test2))

        # identify test cell datapoints
        idx_test1 = np.array(np.where(self.cell_nos == cell_test1)).reshape(-1)
        idx_test2 = np.array(np.where(self.cell_nos == cell_test2)).reshape(-1)
        idx_test = np.hstack([idx_test1, idx_test2]).reshape(-1)

        # identify training cell datapoints
        idx_train = np.delete(np.arange(self.X.shape[0]), idx_test)

        # return train and test datasets
        X_test1 = self.X[idx_test1, :]
        y_test1 = self.y[idx_test1]
        X_test2 = self.X[idx_test2, :]
        y_test2 = self.y[idx_test2]
        X_train = self.X[idx_train, :]
        y_train = self.y[idx_train]

        return X_train, y_train, X_test1, y_test1, X_test2, y_test2, cell_test1, cell_test2

    def train_and_predict(self, X_train, y_train, X_test1, cell_test1, X_test2=None, cell_test2=None,
                          X_test3=None, cell_test3=None, X_test4=None, cell_test4=None):

        n_bootstrap = int(0.9*X_train.shape[0]) # fraction of training set to use to train each model in ensemble
        states = self.n_ensembles*np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed
        dts = '../results/{}'.format(self.experiment)

        y_pred_trs = []
        y_pred_te1s = []
        y_pred_te2s = []
        y_pred_te3s = []
        y_pred_te4s = []

        for i, ensemble_state in enumerate(states):

            # bootstrap from training set and train XGB model
            np.random.seed(ensemble_state)
            idx = np.random.permutation(X_train.shape[0])[0:n_bootstrap]
            regr = xgb.XGBRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=ensemble_state+self.n_split)
            regr.fit(X_train[idx], y_train[idx])

            # save model
            with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test1), 'wb') as f:
                pickle.dump(regr, f)

            # make predictions
            y_pred_tr = regr.predict(X_train)
            y_pred_trs.append(y_pred_tr.reshape(1, y_pred_tr.shape[0], -1))
            y_pred_te1 = regr.predict(X_test1)
            y_pred_te1s.append(y_pred_te1.reshape(1, y_pred_te1.shape[0], -1))

            if X_test2 is not None:
                y_pred_te2 = regr.predict(X_test2)
                y_pred_te2s.append(y_pred_te2.reshape(1, y_pred_te2.shape[0], -1))
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test2), 'wb') as f:
                    pickle.dump(regr, f)

            if X_test3 is not None:
                y_pred_te3 = regr.predict(X_test3)
                y_pred_te3s.append(y_pred_te3.reshape(1, y_pred_te3.shape[0], -1))
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test3), 'wb') as f:
                    pickle.dump(regr, f)

            if X_test4 is not None:
                y_pred_te4 = regr.predict(X_test4)
                y_pred_te4s.append(y_pred_te4.reshape(1, y_pred_te4.shape[0], -1))
                with open('{}/models/{}_{}_{}.pkl'.format(dts, self.experiment_name, i, cell_test4), 'wb') as f:
                    pickle.dump(regr, f)

        # aggregate predictions from each model in ensemble
        y_pred_trs = np.vstack(y_pred_trs)
        y_pred_te1s = np.vstack(y_pred_te1s)
        y_pred_tr = np.mean(y_pred_trs, axis=0)
        y_pred_tr_err = np.sqrt(np.var(y_pred_trs, axis=0))
        y_pred_te1 = np.mean(y_pred_te1s, axis=0)
        y_pred_te1_err = np.sqrt(np.var(y_pred_te1s, axis=0))

        if X_test2 is not None:
            y_pred_te2s = np.vstack(y_pred_te2s)
            y_pred_te2 = np.mean(y_pred_te2s, axis=0)
            y_pred_te2_err = np.sqrt(np.var(y_pred_te2s, axis=0))
        else:
            y_pred_te2 = None
            y_pred_te2_err = None

        if X_test3 is not None:
            y_pred_te3s = np.vstack(y_pred_te3s)
            y_pred_te3 = np.mean(y_pred_te3s, axis=0).reshape(-1)
            y_pred_te3_err = np.sqrt(np.var(y_pred_te3s, axis=0)).reshape(-1)
        else:
            y_pred_te3 = None
            y_pred_te3_err = None

        if X_test4 is not None:
            y_pred_te4s = np.vstack(y_pred_te4s)
            y_pred_te4 = np.mean(y_pred_te4s, axis=0).reshape(-1)
            y_pred_te4_err = np.sqrt(np.var(y_pred_te4s, axis=0)).reshape(-1)
        else:
            y_pred_te4 = None
            y_pred_te4_err = None

        return y_pred_tr.reshape(-1), y_pred_tr_err.reshape(-1), y_pred_te1.reshape(-1), y_pred_te1_err.reshape(-1), y_pred_te2.reshape(-1), y_pred_te2_err.reshape(-1), y_pred_te3, y_pred_te3_err, y_pred_te4, y_pred_te4_err

    def analysis(self, log_name, experiment_info):

        r2s_tr = []
        r2s_te = []
        pes_tr = []
        pes_te = []

        for n_split in range(self.n_splits):
            self.n_split = n_split

            # split data: 2 test cells
            X_train, y_train, X_test1, y_test1, X_test2, y_test2, cell_test1, cell_test2 = self.split_by_cell()

            # train model and predict on train and test data
            y_pred_tr, y_pred_tr_err, y_pred_te1, y_pred_te1_err, y_pred_te2, y_pred_te2_err, _, _, _, _ = self.train_and_predict(X_train, y_train, X_test1, cell_test1=cell_test1, X_test2=X_test2, cell_test2=cell_test2)

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
        r2_tr = np.median(np.array(r2s_tr))
        r2_te = np.median(np.array(r2s_te))
        pe_tr = 100*np.median(np.hstack(pes_tr).reshape(-1))
        pe_te = 100*np.median(np.hstack(pes_te).reshape(-1))
        print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))
        with open(log_name, 'a+') as file:
            file.write(experiment_info)
            file.write('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}\n'.format(r2_tr, pe_tr, r2_te, pe_te))
        return

    def train_no_predict(self, X_train, y_train):

        n_bootstrap = int(0.9*X_train.shape[0]) # fraction of training set to use to train each model in ensemble
        states = self.n_ensembles*np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed
        dts = '../results/{}'.format(self.experiment)

        for i, ensemble_state in enumerate(states):

            # bootstrap from training set and train XGB model
            np.random.seed(ensemble_state)
            idx = np.random.permutation(X_train.shape[0])[0:n_bootstrap]
            regr = xgb.XGBRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=ensemble_state+self.n_split)
            regr.fit(X_train[idx], y_train[idx])

            # save model
            with open('{}/models/{}_{}.pkl'.format(dts, self.experiment_name, i), 'wb') as f:
                pickle.dump(regr, f)

        return

    def analysis_vd2(self, log_name, experiment_info):

        r2s_tr = []
        r2s_te = []
        pes_tr = []
        pes_te = []

        for n_split in range(self.n_splits):
            self.n_split = n_split

            # split data: 2 test cells
            X_train, y_train, X_test1, y_test1, X_test2, y_test2, X_test3, y_test3, X_test4, y_test4, cell_test1, cell_test2, cell_test3, cell_test4 = self.split_into_four()

            # train model and predict on train and test data
            y_pred_tr, y_pred_tr_err, y_pred_te1, y_pred_te1_err, y_pred_te2, y_pred_te2_err, y_pred_te3, y_pred_te3_err, y_pred_te4, y_pred_te4_err = self.train_and_predict(X_train, y_train, X_test1, cell_test1=cell_test1, X_test2=X_test2, cell_test2=cell_test2,
                                                                                                                                                                              X_test3=X_test3, cell_test3=cell_test3, X_test4=X_test4, cell_test4=cell_test4)

            dts = '../results/{}'.format(self.experiment)
            # save test cell predictions
            np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_pred_te1)
            np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_pred_te1_err)
            np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test1), y_test1)
            np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_pred_te2)
            np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_pred_te2_err)
            np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test2), y_test2)

            np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test3), y_pred_te3)
            np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test3), y_pred_te3_err)
            np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test3), y_test3)

            np.save('{}/predictions/pred_mn_{}_{}.npy'.format(dts, self.experiment_name, cell_test4), y_pred_te4)
            np.save('{}/predictions/pred_std_{}_{}.npy'.format(dts, self.experiment_name, cell_test4), y_pred_te4_err)
            np.save('{}/predictions/true_{}_{}.npy'.format(dts, self.experiment_name, cell_test4), y_test4)

            r2s_tr.append(r2_score(y_train, y_pred_tr))
            r2s_te.append(r2_score(y_test1, y_pred_te1))
            r2s_te.append(r2_score(y_test2, y_pred_te2))
            r2s_te.append(r2_score(y_test3, y_pred_te3))
            r2s_te.append(r2_score(y_test4, y_pred_te4))
            pes_tr.append(np.abs(y_train - y_pred_tr) / y_train)
            pes_te.append(np.abs(y_test1 - y_pred_te1) / y_test1)
            pes_te.append(np.abs(y_test2 - y_pred_te2) / y_test2)
            pes_te.append(np.abs(y_test3 - y_pred_te3) / y_test3)
            pes_te.append(np.abs(y_test4 - y_pred_te4) / y_test4)
        r2_tr = np.median(np.array(r2s_tr))
        r2_te = np.median(np.array(r2s_te))
        pe_tr = 100*np.median(np.hstack(pes_tr).reshape(-1))
        pe_te = 100*np.median(np.hstack(pes_te).reshape(-1))
        print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))
        print(r2s_te)
        #print(pes_te)
        with open(log_name, 'a+') as file:
            file.write(experiment_info)
            file.write('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}\n'.format(r2_tr, pe_tr, r2_te, pe_te))
        return
