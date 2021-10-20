# First look at and plot the EIS spectra of the cells over the course of their cycle life to now
import os
import sys
import time

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares
import numpy as np
import pandas as pd


import re
from copy import deepcopy

sys.path.append('../')
import pickle

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
#from utils.feature_util import discharge_features, charge_features, eis_features, check_files
import pdb

experiment_map = {'PJ097':'variable-discharge',
                  'PJ098':'variable-discharge',
                  'PJ099':'variable-discharge',
                  'PJ100':'variable-discharge',
                  'PJ101':'variable-discharge',
                  'PJ102':'variable-discharge',
                  'PJ103':'variable-discharge',
                  'PJ104':'variable-discharge',
                  'PJ105':'variable-discharge',
                  'PJ106':'variable-discharge',
                  'PJ107':'variable-discharge',
                  'PJ108':'variable-discharge',
                  'PJ109':'variable-discharge',
                  'PJ110':'variable-discharge',
                  'PJ111':'variable-discharge',
                  'PJ112':'variable-discharge',
                  'PJ121':'fixed-discharge',
                  'PJ122':'fixed-discharge',
                  'PJ123':'fixed-discharge',
                  'PJ124':'fixed-discharge',
                  'PJ125':'fixed-discharge',
                  'PJ126':'fixed-discharge',
                  'PJ127':'fixed-discharge',
                  'PJ128':'fixed-discharge',
                  'PJ129':'fixed-discharge',
                  'PJ130':'fixed-discharge',
                  'PJ131':'fixed-discharge',
                  'PJ132':'fixed-discharge',
                  'PJ133':'fixed-discharge',
                  'PJ134':'fixed-discharge',
                  'PJ135':'fixed-discharge',
                  'PJ136':'fixed-discharge',
                  'PJ145':'variable-discharge',
                  'PJ146':'variable-discharge',
                  'PJ147':'variable-discharge',
                  'PJ148':'variable-discharge',
                  'PJ149':'variable-discharge',
                  'PJ150':'variable-discharge',
                  'PJ151':'variable-discharge',
                  'PJ152':'variable-discharge',
                  }
column_map = {
    'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
    'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
}


tot_features = []
colors = []

feature_type = 'full'
n_repeats = 1
n_steps = 4
new_log_freq = np.linspace(-1.66, 3.9, 100)

freq1 = np.log10(2.16)
freq2 = np.log10(17.8)
idx_freq1 = np.argmin(np.abs(new_log_freq-freq1))
idx_freq2 = np.argmin(np.abs(new_log_freq-freq2))

nl = 0

column_map = {
    'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
    'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
}

cmap_names = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'RdPu', 'BuPu', 'GnBu', 'YlOrRd']


def plot_trajectories(cells_to_plot, cell_idx, cap_ds, colors, data, experiment_name='variable-discharge'):

    mpl.rcParams['font.family'] = 'sans-serif'
    fontsize = 18
    linewidth = 2.0
    xmin = 0
    xmax = 120
    xlabels = [0, 60, 120]
    y1min = -150
    y1max = 150
    y2min = 0
    y2max = 43
    y1labels = ['-4', '0', '4']
    y1values = [-140, 0, 140]
    y2labels = ['0', '1.2']
    y2values = [0, 42]

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))


    actions = extract_input('actions', data)
    actions[:, 0] *= -1

    for i, (cell, color) in enumerate(zip(cells_to_plot, colors)):
        idx = np.where(cell_idx == cell)
        cell_actions = actions[idx].reshape(-1)
        y = cap_ds[idx].reshape(-1)
        cycles1 = np.arange(0, cell_actions.shape[0], 1) / 3
        cycles2 = np.arange(0, y.shape[0], 1)
        axs[i, 0].scatter(cycles1, cell_actions, c=color)
        axs[i, 1].scatter(cycles2, y, c=color)

    for ax in axs.flat:
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(linewidth)
        for axis in ['top', 'right']:
            ax.spines[axis].set_visible(False)
    axs[-1, 0].set_xlabel('Cycle number', fontsize=fontsize+4)
    axs[1, 0].set_ylabel('Current / 1C-rate', fontsize=fontsize+4)
    axs[-1, 1].set_xlabel('Cycle number', fontsize=fontsize+4)
    axs[1, 1].set_ylabel('Discharge capacity / 1C', fontsize=fontsize+4)

    for ax in axs[:, 0].flat:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y1min, y1max)
        ax.set_yticks(y1values)
        ax.set_yticklabels(y1labels, fontsize=fontsize)

    for ax in axs[:, 1].flat:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y2min, y2max)
        ax.set_yticks(y2values)
        ax.set_yticklabels(y2labels, fontsize=fontsize)

    for ax in axs[:2, :].flat:
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=fontsize)

    for ax in axs[2, :].flat:
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=fontsize)

    fig.tight_layout()
    fig.savefig('figures/{}-examples.png'.format(experiment_name), dpi=400)

def general_sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(b*(x-c)))

def cv_features(capacity, v):
    # Extra feature from discharge curve found by looking at the capacity matrix; difference in discharge curve over
    # the course of cell lifetime.
    # Fit a spline interpolation function to the capacity voltage curve and then return 1000 capacity values evenly
    # spaced between the min and max voltage (following P. Attia's work)
    x = np.linspace(np.min(v), np.max(v), 1000)
    f = interp1d(v, capacity)
    c = f(x)

    return c

def discharge_features(ptf, cycle, cap_curve_norm=None):

    if os.path.isfile(ptf):
        # open txt file
        df = pd.read_csv(ptf, delimiter='\t')
        while len(df.columns) != 7:
            print('Wrong parameters exported for {}. Re-export txt file using rl-gcpl template.'.format(ptf))
            time.sleep(120)
            df = pd.read_csv(ptf, delimiter='\t')

        df.columns = column_map['GCPL']
        dfd = df.loc[(df['i'] < 0) & (df['ox/red']==0)]

        voltage = dfd['ewe'].to_numpy()
        capacity = dfd['capacity'].to_numpy()

        # extract starting (resting) charge voltage, v0
        v0 = np.max(voltage)

        v_mid = 0.5*(np.max(voltage) + np.min(voltage))
        v = voltage - v_mid
        cap = capacity[-1]

        # Initial guess at parameters - helps find the optimal solution
        p0 = [cap, 1.0, 0.01]

        try:
            sigmoid_params, _ = curve_fit(general_sigmoid, v, capacity, p0)
        except RuntimeError:
            sigmoid_params = np.array(p0)

        # extract features from the capacity-voltage discharge curve
        cap_curve = cv_features(capacity, v)

        if cycle == 0:
            log_var = -1.0
            log_iqr = -1.0

        else:
            delta_c = cap_curve - cap_curve_norm
            if np.var(delta_c) <= 1.0e-01:
                log_var = -1.0
                log_iqr = -1.0
            else:
                log_var = np.log10(np.var(delta_c))
                log_iqr = np.log10(iqr(delta_c))

        # extract final (resting) discharge voltage, v1
        t1 = dfd['time'].to_numpy().max()
        dfr = df.loc[df['time']>t1]
        v1 = dfr['ewe'].to_numpy().max()

        # compute discharge rate
        d_rate = dfd['i'].to_numpy()[-10]*-1

        # compute energy outputted
        power = dfd['power'].to_numpy()
        time = dfd['time'].to_numpy()
        e_out = -simps(power, time)


        features = np.hstack([sigmoid_params, np.array([v0, v1, log_var, log_iqr])])

        return cap, features, e_out, d_rate, cap_curve

    else:
        return None

def charge_cap(c1, t1, c2, t2):
    return c1*t1 + c2*t2


def charge_features(ptf):
    """
    Extract time to charge to 80% full capacity and also charge capacity
    :param channel:
    :param path:
    :return:
    """
    # open txt file
    if os.path.isfile(ptf):
        df = pd.read_csv(ptf, delimiter='\t')
        while len(df.columns) != 7:
            print('Wrong parameters exported for {}. Re-export txt file using rl-gcpl template.'.format(ptf))
            time.sleep(120)
            df = pd.read_csv(ptf, delimiter='\t')

        df.columns = column_map['GCPL']
        try:
            ocv = np.mean(df.iloc[0:5].ewe.to_numpy())
        except:
            ocv = None

        # calculate time of 1st and 2nd charging stages
        idx = df.loc[df.i == 0.0].index.to_numpy()
        change = np.array(np.where((idx[1:] - idx[:-1]) != 1)).reshape(-1)
        assert change.shape[0] == 2, "Error in computing time to change"
        idx_start1 = idx[change[0]]
        idx_change1 = idx[change[0]+1]
        idx_start2 = idx[change[1]]
        idx_change2 = idx[change[1]+1]

        t1 = df.iloc[idx_change1].time - df.iloc[idx_start1].time
        t2 = df.iloc[idx_change2-1].time - df.iloc[idx_start2].time
        df_charge = df.loc[df.i > 0.0]
        df1_charge = df.loc[(df.i > 0.0) & (df.time < df.iloc[idx_change1].time)]
        df2_charge = df.loc[(df.i > 0.0) & (df.time > df.iloc[idx_start2].time)]

        t1 = df1_charge.time.max() - df1_charge.time.min()
        t2 = df2_charge.time.max() - df2_charge.time.min()

        c1_rate = np.mean(df1_charge.i.to_numpy())
        c2_rate = np.mean(df2_charge.i.to_numpy())

        cap1 = df1_charge.capacity.max()
        cap2 = df2_charge.capacity.max()
        computed_cap = (c1_rate*t1 + c2_rate*t2)/3600

        t_charge = df_charge.time.max() - df_charge.time.min() - (df.iloc[idx_start2].time - df.iloc[idx_change1].time)
        charge_cap = df_charge.capacity.max()

        #print('{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(cap1, cap2, computed_cap, charge_cap))
        return t_charge / 3600, charge_cap, c1_rate, c2_rate, t1/3600, t2/3600, ocv

    else:
        return None

def overall_fitness_er(p, freq, re_z, im_z):

    (r1, r2, c2, r3, c3, A) = p

    computed_re_z = real_z_er(freq, r1, r2, c2, r3, c3, A)
    computed_im_z = imaginary_z_er(freq, r2, c2, r3, c3, A)

    computed_mod_z = np.sqrt(computed_re_z**2 + computed_im_z**2)
    computed_phase_z = np.arctan(computed_im_z / computed_re_z)
    mod_z = np.sqrt(re_z**2 + im_z**2)
    phase_z = np.arctan(im_z / re_z)

    penalty = (re_z - computed_re_z)**2 + (im_z - computed_im_z)**2 + (mod_z - computed_mod_z)**2 + (phase_z - computed_phase_z)**2

    return penalty

def real_z_er(freq, r1, r2, c2, r3, c3, A):
    w = 2*np.pi*freq

    return r1 + r2 / (1+(w*r2*c2)**2) + (r3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def imaginary_z_er(freq, r2, c2, r3, c3, A):
    w = 2*np.pi*freq

    return w*r2**2*c2 / (1+(w*r2*c2)**2) + (w*c3*(r3+A/w**0.5)**2  + A**2*c3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)


def overall_fitness_r(p, freq, re_z, im_z):

    (r1, r3, c3, A) = p

    computed_re_z = real_z_r(freq, r1, r3, c3, A)
    computed_im_z = imaginary_z_r(freq, r3, c3, A)

    computed_mod_z = np.sqrt(computed_re_z**2 + computed_im_z**2)
    computed_phase_z = np.arctan(computed_im_z / computed_re_z)
    mod_z = np.sqrt(re_z**2 + im_z**2)
    phase_z = np.arctan(im_z / re_z)

    penalty = (re_z - computed_re_z)**2 / (np.mean(re_z))**2 + (im_z - computed_im_z)**2 / (np.mean(im_z))**2+ (mod_z - computed_mod_z)**2 / (np.mean(mod_z))**2 + (phase_z - computed_phase_z)**2 / (np.mean(phase_z))**2

    return penalty

def real_z_r(freq, r1, r3, c3, A):
    w = 2*np.pi*freq

    return r1 + (r3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def imaginary_z_r(freq, r3, c3, A):
    w = 2*np.pi*freq

    return (w*c3*(r3+A/w**0.5)**2  + A**2*c3 + A/w**0.5) / ((1 + A*c3*w**0.5)**2 + (c3*w*(r3 + A/w**0.5))**2)

def extract_features(log_freq, re_z, im_z):
    assert log_freq.shape == re_z.shape == im_z.shape

    freq = 10**log_freq

    popt2, pcov2 = curve_fit(imaginary_z, freq, im_z)
    popt1, pcov1 = curve_fit(real_z, freq, re_z, p0=np.insert(popt2, 0, re_z.min()))
    pdb.set_trace()

    return popt1, popt2

def eis_features(path, feature_type='full', new_log_freq=np.linspace(-1.66, 3.9, 500), n_repeats=1):
    # Get initial features of discharge EIS spectrum
    df = pd.read_csv(path, delimiter='\t')
    df.columns = column_map['GEIS']

    re_z = df['re_z'].to_numpy()
    im_z = df['-im_z'].to_numpy()
    freq = df['freq'].to_numpy()

    re_z = np.mean(re_z.reshape(n_repeats, -1), axis=0)
    im_z = np.mean(im_z.reshape(n_repeats, -1), axis=0)
    freq = np.mean(freq.reshape(n_repeats, -1), axis=0)
    log_freq = np.log10(freq)

    # interpolate
    f1 = interp1d(log_freq, re_z, kind='cubic')
    f2 = interp1d(log_freq, im_z, kind='cubic')

    # compute new values
    re_z = f1(new_log_freq)
    im_z = f2(new_log_freq)

    # extract parameters from re, im
    if feature_type == 'extended-randles':
        ls = least_squares(overall_fitness_er, x0=(1, 0.1, 0.1, 1, 1, 0.3), bounds=([0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 100, 100]), args=(10**new_log_freq, re_z, im_z))
        x1 = ls.x.reshape(-1)

    elif feature_type == 'randles':
        ls = least_squares(overall_fitness_r, x0=(1, 0.1, 0.1, 0.3), bounds=([0, 0, 0, 0], [100, 100, 100, 100]), args=(10**new_log_freq, re_z, im_z))
        x1 = ls.x.reshape(-1)

    else:
        x1 = np.hstack((re_z, im_z)).reshape(-1)

    return x1

def check_files(ptf_files):
    for ptf in ptf_files:
        if not os.path.isfile(ptf):
            print(ptf)
        else:
            continue
    return



def train_test_split(X, y, test_fraction=0.2, seed=10):
    n = X.shape[0]
    np.random.seed(seed)
    idx = np.random.permutation(n)
    idx_te = idx[0:int(test_fraction*n)]
    idx_tr = idx[int(test_fraction * n):]
    return X[idx_tr, :], y[idx_tr], X[idx_te, :], y[idx_te], idx_tr, idx_te

import forestci as fci

class OneDimRFRegressor:

    def __init__(self, X, y, cell_nos, experiment, n_ensembles=10, n_splits=5, start_seed=1):

        np.random.seed(start_seed+10)
        self.start_seed = start_seed
        self.n_ensembles = n_ensembles
        self.n_splits = n_splits
        self.n_split = 0
        self.X = X
        self.y = y
        self.cell_nos = cell_nos
        self.n = X.shape[0] # number of datapoints
        self.n_te = self.n // n_splits + 1 # number of test points
        self.idx = np.random.permutation(self.n)
        self.cell_idx = np.unique(self.cell_nos)
        self.experiment = experiment
        print(self.experiment)
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

    def split_by_cell(self):

        idx_te = np.array(np.where(self.cell_nos == self.cell_idx[self.n_split])).reshape(-1)
        cellno = (int(self.n_splits + self.n_split)%len(self.cell_idx))
        cellno2 = (int(2*self.n_splits + self.n_split)%len(self.cell_idx))
        print('{} {} {}'.format(self.n_split, cellno, cellno2))
        idx_tr = np.delete(np.arange(self.X.shape[0]), idx_te)
        X_test = self.X[idx_te, :]
        X_train = self.X[idx_tr, :]
        y_test = self.y[idx_te]
        y_train = self.y[idx_tr]

        return X_train, y_train, X_test, y_test

    def split_data(self):
        if self.n_split == (self.n_splits-1):
            idx_te = self.idx[self.n_split * self.n_te:]
            idx_tr = self.idx[0:self.n_split * self.n_te]
        else:
            idx_te = self.idx[self.n_split * self.n_te: (self.n_split + 1) * self.n_te]
            idx_tr = np.delete(self.idx, np.arange(self.n_split * self.n_te, (self.n_split + 1) * self.n_te))

        X_test = self.X[idx_te, :]
        X_train = self.X[idx_tr, :]
        y_test = self.y[idx_te]
        y_train = self.y[idx_tr]

        return X_train, y_train, X_test, y_test

    def train_and_predict(self, X_train, y_train, X_test, max_depth=25, n_estimators=1000, save_model=False):
        regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        regr.fit(X_train, y_train)

        # Make predictions
        y_pred_tr = regr.predict(X_train)
        y_pred_tr_err = np.sqrt(fci.random_forest_error(regr, X_train, X_train))
        y_pred_te = regr.predict(X_test)
        y_pred_te_var = np.sqrt(fci.random_forest_error(regr, X_train, X_test))

        """
        states = self.n_ensembles*np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed

        y_pred_trs = []
        y_pred_tes = []

        for i, ensemble_state in enumerate(states):
            regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=ensemble_state)
            regr.fit(X_train, y_train)

            # Make predictions
            y_pred_tr = regr.predict(X_train)
            y_pred_tr_var = fci.random_forest_error(regr, X_train, X_train)
            y_pred_trs.append(y_pred_tr.reshape(1, y_pred_tr.shape[0], -1))
            y_pred_te = regr.predict(X_test)

            y_pred_tes.append(y_pred_te.reshape(1, y_pred_te.shape[0], -1))

            if save_model:
                with open('processed/models/{}/rf_{}_{}.pkl'.format(self.experiment, self.cell_idx[self.n_split], i), 'wb') as f:
                    pickle.dump(regr, f)
        y_pred_trs = np.vstack(y_pred_trs)
        y_pred_tes = np.vstack(y_pred_tes)
        y_pred_tr = np.mean(y_pred_trs, axis=0)
        y_pred_tr_err = np.sqrt(np.var(y_pred_trs, axis=0))
        y_pred_te = np.mean(y_pred_tes, axis=0)
        y_pred_te_err = np.sqrt(np.var(y_pred_tes, axis=0))
        """
        if save_model:
            with open('processed/models/{}/rf-fci_{}.pkl'.format(self.experiment, self.cell_idx[self.n_split]), 'wb') as f:
                pickle.dump(regr, f)

        return y_pred_tr.reshape(-1), y_pred_tr_err.reshape(-1), y_pred_te.reshape(-1), y_pred_te_err.reshape(-1)

    def train_model(self, X_train, y_train, experiment_name, max_depth=25, n_estimators=1000):
        regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        regr.fit(X_train, y_train)
        with open('processed/{}/models/rf-fci_{}.pkl'.format(self.experiment, experiment_name), 'wb') as f:
            pickle.dump(regr, f)
        """
        states = self.n_ensembles*np.arange(1, self.n_ensembles + 1, 1) + self.n_split + self.start_seed

        for i, ensemble_state in enumerate(states):
            regr = RandomForestRegressor(max_depth=max_depth, random_state=ensemble_state)
            regr.fit(X_train, y_train)

            with open('processed/{}/models/rf_{}_{}.pkl'.format(self.experiment, experiment_name, i), 'wb') as f:
                pickle.dump(regr, f)
        """
        return


    def analysis(self, experiment_info, experiment_name, log_name='log-fixed-discharge.txt', xmin=0, xmax=800, fontsize=14, by_cell=True, max_depth=10,
                 pca_components=None, step=1, save_predictions=True):

        subdir = self.experiment
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        fig_te, ax_te = plt.subplots(1, 1, figsize=(7, 7))

        r2s_tr = []
        r2s_te = []
        rmses_tr = []
        rmses_te = []
        rmses_be = []

        for n_split in range(self.n_splits):
            self.n_split = n_split
            fname = 'PJ{}'.format(self.cell_idx[self.n_split])
            # Split data
            if by_cell:
                X_train, y_train, X_test, y_test = self.split_by_cell()
            else:
                X_train, y_train, X_test, y_test = self.split_data()

            # Train model and predict on train and test data
            y_pred_tr, y_pred_tr_err, y_pred_te, y_pred_te_err = self.train_and_predict(X_train, y_train, X_test, max_depth=max_depth)
            bench = np.mean(y_train) * np.ones(y_pred_te.shape)

            np.save('processed/{}/pred_mn_{}_{}.npy'.format(subdir, experiment_name, fname), y_pred_te)
            np.save('processed/{}/pred_std_{}_{}.npy'.format(subdir, experiment_name, fname), y_pred_te_err)
            np.save('processed/{}/true_{}_{}.npy'.format(subdir, experiment_name, fname), y_test)

            ax_te.errorbar(y_test, y_pred_te, yerr=y_pred_te_err, marker='o', linestyle='', capsize=2.0,
                        label='Split {}'.format(self.n_split + 1))
            ax.errorbar(y_train, y_pred_tr, yerr=y_pred_tr_err, marker='o', linestyle='', capsize=2.0,
                        label='Split {}'.format(self.n_split + 1))

            r2s_tr.append(r2_score(y_train, y_pred_tr))
            r2s_te.append(r2_score(y_test, y_pred_te))
            rmses_be.append(np.sqrt(mean_squared_error(y_test, bench)))
            rmses_tr.append(np.sqrt(mean_squared_error(y_train, y_pred_tr)))
            rmses_te.append(np.sqrt(mean_squared_error(y_test, y_pred_te)))

        ax.plot(np.linspace(xmin, xmax, 4), np.linspace(xmin, xmax, 4), '-.', c='grey')
        ax_te.plot(np.linspace(xmin, xmax, 4), np.linspace(xmin, xmax, 4), '-.', c='grey')
        ax.set_xlabel('True', fontsize=fontsize)
        ax.set_ylabel('Predicted', fontsize=fontsize)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax_te.set_xlim(xmin, xmax)
        ax_te.set_ylim(xmin, xmax)
        ax_te.set_xlabel('True', fontsize=fontsize)
        ax_te.set_ylabel('Predicted', fontsize=fontsize)
        ax_te.legend(fontsize=fontsize, ncol=2)
        ax.legend(fontsize=fontsize, ncol=2)

        r2s_te = np.array(r2s_te)
        r2s_tr = np.array(r2s_tr)
        rmses_tr = np.array(rmses_tr)
        rmses_te = np.array(rmses_te)
        rmses_be = np.array(rmses_be)

        print(r2s_te)

        print('Train RMSE (median):{}\t RMSE (mean):{}\t R2 (median):{}'.format(np.median(rmses_tr), np.mean(rmses_tr), np.median(r2s_tr)))
        print('Test RMSE (median):{}\t RMSE (mean):{}\t R2 (median):{}'.format(np.median(rmses_te), np.mean(rmses_te), np.median(r2s_te)))
        print('Bench RMSE (median):{}\t RMSE (mean):{}'.format(np.median(rmses_be), np.mean(rmses_be)))

        with open(log_name, 'a+') as file:
            file.write(experiment_info)
            file.write('Train RMSE (median):{:.4f}\t RMSE (mean):{:.4f}\t R2 (median):{:.4f}\n'.format(np.median(rmses_tr), np.mean(rmses_tr), np.median(r2s_tr)))
            file.write('Test RMSE (median):{:.4f}\t RMSE (mean):{:.4f}\t R2 (median):{:.4f}\n'.format(np.median(rmses_te), np.mean(rmses_te), np.median(r2s_te)))
            file.write('Bench RMSE (median):{:.4f}\t RMSE (mean):{:.4f}\n'.format(np.median(rmses_be), np.mean(rmses_be)))

        return

    def protocol_analysis(self, n_split=0):
        self.n_split = n_split
        X_train, y_train, X_test, y_test = self.split_by_cell()
        fig, ax = plt.subplots(1, 1)
        fontsize = 14

        c1s = []
        c2s = []
        ds = []

        for i in range(X_train.shape[0]):
            c1, c2, d1 = optimal_protocol(X_train[i], self.cell_idx[self.n_split])
            c1s.append(c1)
            c2s.append(c2)
            ds.append(d1)
        c1s = np.array(c1s)
        c2s = np.array(c2s)
        ds = np.array(ds)
        cmap = plt.get_cmap('YlOrRd_r')
        scatter_plot = ax.scatter(c1s, c2s, c=ds, cmap=cmap)
        cbar = plt.colorbar(scatter_plot)
        ax.set_xlabel('Charging rate (stage 1) / mA', fontsize=fontsize)
        ax.set_ylabel('Charging rate (stage 2) / mA', fontsize=fontsize)
        cbar.set_label('Discharge rate / mA', fontsize=fontsize, rotation=270)
        plt.show()
        return


def ensemble_predict(x, cell_id, n_ensembles=10):
    y_preds = []
    for i in range(n_ensembles):
        with open('processed/2cc/rf{}_{}.pkl'.format(cell_id, i), 'rb') as f:
            regr = pickle.load(f)
            y_pred = regr.predict(x)
            y_preds.append(y_pred.reshape(1, y_pred.shape[0], -1))
    y_preds = np.vstack(y_preds)
    y_pred = np.mean(y_preds, axis=0)
    y_pred_err = np.sqrt(np.var(y_preds, axis=0))
    return y_pred.reshape(-1), y_pred_err.reshape(-1)

def optimal_protocol(x, cell_id):

    nx = 50
    ny = 50
    cx, cy = np.meshgrid(np.linspace(70.0, 140.0, nx), np.linspace(35.0, 105.0, ny))

    y_true_pred, y_true_pred_err = ensemble_predict(x.reshape(1, -1), cell_id)

    x_start = x.reshape(1, -1)[:, :-2]
    x_end = np.concatenate((cx.reshape((-1, 1)), cy.reshape((-1, 1))), axis=1)
    x_expanded = np.concatenate((np.tile(x_start.reshape(-1), (nx*ny, 1)), x_end), axis=1)
    y_pred, y_pred_err = ensemble_predict(x_expanded, cell_id)
    #print('Max: {:.2f}\tMin: {:.2f}\tTrue: {:.2f}\tTrue (pred){:.2f}+-{:.2f}'.format(np.max(y_pred), np.min(y_pred), y_true, y_true_pred[0], y_true_pred_err[0]))
    y_pred = y_pred.reshape((nx, ny))
    y_pred_err = y_pred_err.reshape((nx, ny))
    id_max = np.where(y_pred == np.max(y_pred))
    #print('Optimal protocol:\t{:.1f}\t{:.1f}'.format())
    return cx[id_max][0], cy[id_max][0], x[-3]

def analyse_protocol(x, y_true, cell_id):

    nx = 50
    ny = 50
    cx, cy = np.meshgrid(np.linspace(70.0, 140.0, nx), np.linspace(35.0, 105.0, ny))

    y_true_pred, y_true_pred_err = ensemble_predict(x.reshape(1, -1), cell_id)

    x_start = x.reshape(1, -1)[:, :-2]
    x_end = np.concatenate((cx.reshape((-1, 1)), cy.reshape((-1, 1))), axis=1)
    x_expanded = np.concatenate((np.tile(x_start.reshape(-1), (nx*ny, 1)), x_end), axis=1)
    y_pred, y_pred_err = ensemble_predict(x_expanded, cell_id)
    print('Max: {:.2f}\tMin: {:.2f}\tTrue: {:.2f}\tTrue (pred){:.2f}+-{:.2f}'.format(np.max(y_pred), np.min(y_pred), y_true, y_true_pred[0], y_true_pred_err[0]))
    y_pred = y_pred.reshape((nx, ny))
    y_pred_err = y_pred_err.reshape((nx, ny))
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('YlOrRd_r')
    heatmap = ax.pcolor(cx, cy, y_pred, cmap=cmap)

    cbar = plt.colorbar(heatmap)
    #cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar.set_label('Predicted capacity', rotation=270)
    ax.annotate('{:.1f}'.format(y_true), (x[-2], x[-1]))

    id_max = np.where(y_pred == np.max(y_pred))

    print('Optimal protocol:\t{:.1f}\t{:.1f}'.format(cx[id_max][0], cy[id_max][0]))

    plt.show()

    fig2, ax2 = plt.subplots()
    heatmap2 = ax2.pcolor(cx, cy, y_pred_err, cmap=cmap)
    cbar2 = plt.colorbar(heatmap2)
    #cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar2.set_label('Predicted capacity err', rotation=270)
    ax2.annotate('{:.1f}'.format(y_true), (x[-2], x[-1]))
    plt.show()

    return

def analyse_global_protocol(d_rate, cell_id):

    nx = 50
    ny = 50
    cx, cy = np.meshgrid(np.linspace(70.0, 140.0, nx), np.linspace(35.0, 105.0, ny))

    x_end = np.concatenate((cx.reshape((-1, 1)), cy.reshape((-1, 1))), axis=1)
    x_expanded = np.concatenate((np.repeat(d_rate, nx*ny).reshape(-1, 1), x_end), axis=1)
    y_pred, y_pred_err = ensemble_predict(x_expanded, cell_id)
    print('Max: {:.2f}\tMin: {:.2f}'.format(np.max(y_pred), np.min(y_pred)))
    y_pred = y_pred.reshape((nx, ny))
    y_pred_err = y_pred_err.reshape((nx, ny))
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('YlOrRd_r')
    heatmap = ax.pcolor(cx, cy, y_pred, cmap=cmap)

    cbar = plt.colorbar(heatmap)
    #cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar.set_label('Predicted capacity', rotation=270)

    id_max = np.where(y_pred == np.max(y_pred))

    print('Optimal protocol:\t{:.1f}\t{:.1f}'.format(cx[id_max][0], cy[id_max][0]))

    plt.show()

    fig2, ax2 = plt.subplots()
    heatmap2 = ax2.pcolor(cx, cy, y_pred_err, cmap=cmap)
    cbar2 = plt.colorbar(heatmap2)
    #cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar2.set_label('Predicted capacity err', rotation=270)
    plt.show()

    return


def ensemble_predict(x, cell_id, n_ensembles=10):
    y_preds = []
    for i in range(n_ensembles):
        with open('processed/2cc/rf_fixed_{}_{}.pkl'.format(cell_id, i), 'rb') as f:
            regr = pickle.load(f)
            y_pred = regr.predict(x)
            y_preds.append(y_pred.reshape(1, y_pred.shape[0], -1))
    y_preds = np.vstack(y_preds)
    y_pred = np.mean(y_preds, axis=0)
    y_pred_err = np.sqrt(np.var(y_preds, axis=0))
    return y_pred.reshape(-1), y_pred_err.reshape(-1)

def identify_cells(experiment):
    if experiment == 'variable-discharge':
        cell_map = {1:['PJ097'],
                     2:['PJ098',],
                     3:['PJ099',],
                     4:['PJ100',],
                     5:['PJ101',],
                     6:['PJ102',],
                     7:['PJ103',],
                     8:['PJ104',]}
        """
        cell_map = {1:['PJ097','PJ105','PJ145'],
                     2:['PJ098','PJ106','PJ146'],
                     3:['PJ099','PJ107','PJ147'],
                     4:['PJ100','PJ108','PJ148'],
                     5:['PJ101','PJ109','PJ149'],
                     6:['PJ102','PJ110','PJ150'],
                     7:['PJ103','PJ111','PJ151'],
                     8:['PJ104','PJ112','PJ152']}
        """

    elif experiment == 'fixed-discharge':
        cell_map = {1:['PJ121', 'PJ129'],
                2:['PJ122', 'PJ130'],
                3:['PJ125', 'PJ131'],
                4:['PJ126', 'PJ132'],
                5:['PJ123', 'PJ133'],
                6:['PJ124', 'PJ134'],
                7:['PJ127', 'PJ135'],
                8:['PJ128', 'PJ136'],}
        """
        cell_map = {1:['PJ121'],
                2:['PJ122'],
                5:['PJ123'],
                6:['PJ124'],
                3:['PJ125'],
                4:['PJ126'],
                7:['PJ127'],
                8:['PJ128'],}
        """

    elif experiment == 'both':
        cell_map = {1:['PJ097','PJ105', 'PJ121', 'PJ129', 'PJ145'],
                2:['PJ098','PJ106', 'PJ122', 'PJ130', 'PJ146'],
                3:['PJ099','PJ107', 'PJ125', 'PJ131', 'PJ147'],
                4:['PJ100','PJ108','PJ126', 'PJ132', 'PJ148'],
                5:['PJ101','PJ109','PJ123', 'PJ133', 'PJ149'],
                6:['PJ102','PJ110','PJ124', 'PJ134', 'PJ150'],
                7:['PJ103','PJ111','PJ127', 'PJ135', 'PJ151'],
                8:['PJ104','PJ112','PJ128', 'PJ136', 'PJ152'],}
    elif experiment == 'both-variable':
        cell_map = {1:['PJ097','PJ105', 'PJ121', 'PJ129', 'PJ145'],
                2:['PJ098','PJ106', 'PJ122', 'PJ130', 'PJ146'],
                3:['PJ099','PJ107', 'PJ125', 'PJ131', 'PJ147'],
                4:['PJ100','PJ108','PJ126', 'PJ132', 'PJ148'],
                5:['PJ101','PJ109','PJ123', 'PJ133', 'PJ149'],
                6:['PJ102','PJ110','PJ124', 'PJ134', 'PJ150'],
                7:['PJ103','PJ111','PJ127', 'PJ135', 'PJ151'],
                8:['PJ104','PJ112','PJ128', 'PJ136', 'PJ152'],}

    else:
        cell_map = None

    return cell_map



def extract_input(input_name, data):
    (c, soh, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates) = data

    actions = np.hstack([d_rates.reshape(-1, 1), c1_rates.reshape(-1, 1), c2_rates.reshape(-1, 1)])

    if input_name == 'eis-cvfs-ct-actions':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)
    elif input_name == 'eis-cvfs-ct-c-actions':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        states = np.concatenate((states, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-cvfs-actions':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-ct-actions':
        states = np.concatenate((eis_ds, cap_throughputs.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-actions':
        states = eis_ds
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'cvfs-actions':
        states = cvfs
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ct-actions':
        states = cap_throughputs.reshape(-1, 1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'cvfs-ct-actions':
        states = np.concatenate((cvfs, cap_throughputs.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-cvfs-ct':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        x = states

    elif input_name == 'eis-cvfs':
        states = np.concatenate((eis_ds, cvfs), axis=1)
        x = states

    elif input_name == 'actions':
        x = actions

    elif input_name == 'eis':
        x = eis_ds

    elif input_name == 'cvfs':
        x = cvfs

    elif input_name == 'ocv':
        x = ocvs.reshape(-1, 1)

    elif input_name == 'ct':
        x = cap_throughputs.reshape(-1, 1)

    elif input_name == 'c':
        x = c.reshape(-1, 1)

    elif input_name == 'soh':
        x = soh.reshape(-1, 1)

    elif input_name == 'c-actions':
        states = c.reshape(-1, 1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'soh-actions':
        states = soh.reshape(-1, 1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-c-actions':
        states = np.concatenate((eis_ds, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'eis-soh-actions':
        states = np.concatenate((eis_ds, soh.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    else:
        print('Choose different input name')
        x = None

    return x

def metrics_calculator(true, pred):
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return r2, rmse


def make_plot(experiment_name, fnames, experiment='variable-discharge'):
    from matplotlib.cm import ScalarMappable
    figsize = (8, 7)
    alpha = 0.65
    vmin = 0
    #vmax = 0.6
    vmax = 0.8
    cmap_type = 'YlOrRd_r'
    xmin = 5
    xmax = 45
    fontsize = 18
    linewidth = 2.0
    plt.rcParams["font.family"] = "Times New Roman"
    subdir = experiment
    pts = 'figures/{}/{}.png'.format(subdir, experiment_name)

    cmap = plt.get_cmap('YlOrRd_r')
    norm = plt.Normalize(vmin, vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sm =  ScalarMappable(norm=norm, cmap=cmap)
    r2s = []
    rmses = []


    for id_te in range(len(fnames)):
        fname = fnames[id_te]

        pred_te_mn = np.load('processed/{}/pred_mn_{}_PJ{}.npy'.format(subdir, experiment_name, fname))
        pred_te_std = np.load('processed/{}/pred_std_{}_PJ{}.npy'.format(subdir, experiment_name, fname))
        y_test = np.load('processed/{}/true_{}_PJ{}.npy'.format(subdir, experiment_name, fname))

        print(pred_te_std.max())
        r2, rmse = metrics_calculator(y_test, pred_te_mn)
        r2s.append(r2)
        rmses.append(rmse)
        ax.scatter(y_test, pred_te_mn, c=pred_te_std, cmap=cmap_type, marker='o', alpha=alpha, vmin=vmin, vmax=vmax)
    labels = [5, 15, 25, 35, 45]
    ticklabels = [0, 0.2, 0.4, 0.6, 0.8]
    #ticklabels = [0, 0.2, 0.4, 0.6]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(ticklabels)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label('Predicted error / mAh', fontsize=fontsize+4)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.plot(np.linspace(xmin, xmax-0.2, 4), np.linspace(xmin, xmax-0.2, 4), '-.', linewidth=linewidth, c='grey')
    ax.set_xlabel('Actual Capacity / mAh', fontsize=fontsize+4)
    ax.set_ylabel('Predicted Capacity / mAh', fontsize=fontsize+4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    fig.savefig(pts, dpi=400)

    r2s = np.array(r2s)
    rmses = np.array(rmses)
    print('R2 scores:')
    print(r2s)
    print('RMSEs:')
    print(rmses)
    print('RMSE (median):{}\t RMSE (mean):{}\t R2 (median):{}'.format(np.median(rmses), np.mean(rmses), np.median(r2s)))

    r2_median = np.median(r2s)
    r2_mn = np.mean(r2s)
    r2_std = np.std(r2s)
    rmse_median = np.median(rmses)
    rmse_mn = np.mean(rmses)
    rmse_std = np.std(rmses)

    return r2_median, r2_mn, r2_std, rmse_median, rmse_mn, rmse_std

def parity_plot(true, pred, pred_err, pts):
    from matplotlib.cm import ScalarMappable
    figsize = (8, 7)
    alpha = 0.65
    vmin = 0
    #vmax = 0.6
    vmax = 0.8
    cmap_type = 'YlOrRd_r'
    xmin = 5
    xmax = 45
    fontsize = 18
    linewidth = 2.0
    plt.rcParams["font.family"] = "Times New Roman"
    subdir = experiment

    cmap = plt.get_cmap('YlOrRd_r')
    norm = plt.Normalize(vmin, vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sm =  ScalarMappable(norm=norm, cmap=cmap)
    r2s = []
    rmses = []
    ax.scatter(true, pred, c=pred_err, cmap=cmap_type, marker='o', alpha=alpha, vmin=vmin, vmax=vmax)
    labels = [5, 15, 25, 35, 45]
    ticklabels = [0, 0.2, 0.4, 0.6, 0.8]
    #ticklabels = [0, 0.2, 0.4, 0.6]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(ticklabels)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label('Predicted error / mAh', fontsize=fontsize+4)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.plot(np.linspace(xmin, xmax-0.2, 4), np.linspace(xmin, xmax-0.2, 4), '-.', linewidth=linewidth, c='grey')
    ax.set_xlabel('Actual Capacity / mAh', fontsize=fontsize+4)
    ax.set_ylabel('Predicted Capacity / mAh', fontsize=fontsize+4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)


    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mpe = np.mean(np.abs(true - pred) / true) * 100
    med_pe = np.median(np.abs(true - pred) / true) * 100
    print('R2 = {}\t RMSE = {}\t MPE = {}\t Med_PE = {}'.format(r2, rmse, mpe, med_pe))
    fig.savefig(pts, dpi=400)

    return

def cc_plot(true, pred, pred_err, pts, metric='rmse', figsize=(7, 7), linewidth=3.0, fontsize=24):
    """
    Plot confidence curve.
    :param mean:
    :param variance:
    :param target:
    :param filename:
    :return:
    """

    conf_percentile, metric_model, metric_oracle = metric_ordering(pred, pred**2, true, metric)

    mpl.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.plot(conf_percentile, metric_oracle, color="C0", linestyle=linestyles['densely dashed'],
            linewidth=linewidth, label="Oracle")
    ax.plot(conf_percentile, metric_model, color="C1", linestyle=linestyles['densely dotted'],
            linewidth=linewidth, label="Model")

    ymin = min(np.min(metric_oracle), np.min(metric_model))
    ymax = max(np.max(metric_oracle), np.max(metric_model))

    ax.set_ylim(ymin, ymax)

    #yticks = [0, 4, 8, 12]

    #yticks = np.arange(np.round(ymin, decimals=1), np.round(ymax + 0.2, decimals=1), step=0.2)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(np.round(yticks, decimals=1), fontsize=fontsize)

    xticks = np.linspace(0, 100, 5)
    xticklabels = [0, 25, 50, 75, 100]
    ax.set_xlim(0, 100)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    ax.set_xlabel("Percentage data imputed (%)", fontsize=fontsize)
    if metric == 'rmse':
        ax.set_ylabel("RMSE", fontsize=fontsize)
    elif metric == 'r2':
        ax.set_ylabel("R2 score", fontsize=fontsize)
    elif metric == 'rmae':
        ax.set_ylabel("Relative MAE (%)", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(pts, dpi=400, transparent=True)
    return



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def metric_ordering(mean, variance, target, metric='rmse'):
    assert len(mean) == len(variance)
    assert len(mean) == len(target)

    if metric == 'rmse':
        n_min = 5
    elif metric == 'r2':
        n_min = 10
    elif metric == 'rmae':
        n_min = 5

    # Actual error
    errors = np.absolute(mean - target)

    metric_model = np.zeros(len(target) - n_min)
    metric_oracle = np.zeros(len(target) - n_min)
    conf_percentile = np.linspace(100, 100 * n_min / (len(target)), len(target) - n_min)

    mean_model = copy.deepcopy(mean)
    mean_oracle = copy.deepcopy(mean)
    target_model = copy.deepcopy(target)
    target_oracle = copy.deepcopy(target)

    for i in range(len(mean) - n_min):
        # Order values according to level of uncertainty
        idx_model = variance.argmax()
        idx_oracle = errors.argmax()

        # Compute the metric using our predictions, using only the X% most confident prediction.
        # The metric should systematically go down (RMSE) or up (R2) as X decreases.
        if metric == 'rmse':
            metric_model[i] = np.sqrt(mean_squared_error(target_model, mean_model))
            metric_oracle[i] = np.sqrt(mean_squared_error(target_oracle, mean_oracle))
        elif metric == 'r2':
            metric_model[i] = r2_score(target_model, mean_model)
            metric_oracle[i] = r2_score(target_oracle, mean_oracle)
        elif metric == 'rmae':
            metric_model[i] = rmae(target_model, mean_model)
            metric_oracle[i] = rmae(target_oracle, mean_oracle)
        else:
            raise Exception('Metric should be rmse, rmae or r2.')

        # Remove least confident prediction of model
        target_model = np.delete(target_model, idx_model)
        mean_model = np.delete(mean_model, idx_model)
        variance = np.delete(variance, idx_model)

        # Remove least confident prediction of oracle
        target_oracle = np.delete(target_oracle, idx_oracle)
        mean_oracle = np.delete(mean_oracle, idx_oracle)
        errors = np.delete(errors, idx_oracle)

    return conf_percentile, metric_model, metric_oracle



def single_confidence_curve(experiment_name, fnames, experiment='variable-discharge', metric='rmse',
                            figsize=(7, 7), linewidth=3.0, fontsize=24):
    """
    Plot confidence curve.
    :param mean:
    :param variance:
    :param target:
    :param filename:
    :return:
    """

    pts = 'figures/{}/{}_conf_curve.png'.format(experiment, experiment_name)

    mean = []
    variance = []
    target = []

    for id_te in range(len(fnames)):
        fname = fnames[id_te]

        pred_te_mn = np.load('processed/{}/pred_mn_{}_PJ{}.npy'.format(experiment, experiment_name, fname))
        pred_te_std = np.load('processed/{}/pred_std_{}_PJ{}.npy'.format(experiment, experiment_name, fname))
        y_test = np.load('processed/{}/true_{}_PJ{}.npy'.format(experiment, experiment_name, fname))

        mean.append(pred_te_mn)
        variance.append(pred_te_std**2)
        target.append(y_test)

    mean = np.hstack(mean)
    variance = np.hstack(variance)
    target = np.hstack(target)

    conf_percentile, metric_model, metric_oracle = metric_ordering(mean, variance, target, metric='rmse')

    mpl.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.plot(conf_percentile, metric_oracle, color="C0", linestyle=linestyles['densely dashed'],
            linewidth=linewidth, label="Oracle")
    ax.plot(conf_percentile, metric_model, color="C1", linestyle=linestyles['densely dotted'],
            linewidth=linewidth, label="Model")

    ymin = min(np.min(metric_oracle), np.min(metric_model))
    ymax = max(np.max(metric_oracle), np.max(metric_model))

    ax.set_ylim(ymin, ymax)

    #yticks = [0, 4, 8, 12]

    #yticks = np.arange(np.round(ymin, decimals=1), np.round(ymax + 0.2, decimals=1), step=0.2)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(np.round(yticks, decimals=1), fontsize=fontsize)

    xticks = np.linspace(0, 100, 5)
    xticklabels = [0, 25, 50, 75, 100]
    ax.set_xlim(0, 100)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    ax.set_xlabel("Percentage data imputed (%)", fontsize=fontsize)
    if metric == 'rmse':
        ax.set_ylabel("RMSE", fontsize=fontsize)
    elif metric == 'r2':
        ax.set_ylabel("R2 score", fontsize=fontsize)
    elif metric == 'rmae':
        ax.set_ylabel("Relative MAE (%)", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(pts, dpi=400, transparent=True)
    return

def extract_data(experiment, channels):

    cell_map = identify_cells(experiment)
    feature_type = 'full'
    n_repeats = 1
    n_steps = 4
    new_log_freq = np.linspace(-1.66, 3.9, 100)

    freq1 = np.log10(2.16)
    freq2 = np.log10(17.8)
    idx_freq1 = np.argmin(np.abs(new_log_freq-freq1))
    idx_freq2 = np.argmin(np.abs(new_log_freq-freq2))

    nl = 0

    c1_rates = []
    c2_rates = []
    d_rates = []
    eis_cs = []
    eis_ds = []
    t_charges = []
    t1_charges = []
    t2_charges = []
    ocvs = []
    cap_cs = []
    cap_ds = []
    cap_nets = []
    cap_throughputs = []
    cap_inits = []
    eis_inits = []
    cycles = []
    cell_idx = []
    cvfs = []
    last_caps = []
    sohs = []

    column_map = {
        'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
        'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
    }

    geis_discharge_no = 1
    gcpl_charge_no = 2
    geis_charge_no = 3
    gcpl_discharge_no = 4

    cmap_names = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'RdPu', 'BuPu', 'GnBu', 'YlOrRd']

    experiment_map = {'PJ097':'variable-discharge',
                      'PJ098':'variable-discharge',
                      'PJ099':'variable-discharge',
                      'PJ100':'variable-discharge',
                      'PJ101':'variable-discharge',
                      'PJ102':'variable-discharge',
                      'PJ103':'variable-discharge',
                      'PJ104':'variable-discharge',
                      'PJ105':'variable-discharge',
                      'PJ106':'variable-discharge',
                      }
    """
    experiment_map = {'PJ097':'variable-discharge',
                      'PJ098':'variable-discharge',
                      'PJ099':'variable-discharge',
                      'PJ100':'variable-discharge',
                      'PJ101':'variable-discharge',
                      'PJ102':'variable-discharge',
                      'PJ103':'variable-discharge',
                      'PJ104':'variable-discharge',
                      'PJ105':'variable-discharge',
                      'PJ106':'variable-discharge',
                      'PJ107':'variable-discharge',
                      'PJ108':'variable-discharge',
                      'PJ109':'variable-discharge',
                      'PJ110':'variable-discharge',
                      'PJ111':'variable-discharge',
                      'PJ112':'variable-discharge',
                      'PJ121':'fixed-discharge',
                      'PJ122':'fixed-discharge',
                      'PJ123':'fixed-discharge',
                      'PJ124':'fixed-discharge',
                      'PJ125':'fixed-discharge',
                      'PJ126':'fixed-discharge',
                      'PJ127':'fixed-discharge',
                      'PJ128':'fixed-discharge',
                      'PJ129':'fixed-discharge',
                      'PJ130':'fixed-discharge',
                      'PJ131':'fixed-discharge',
                      'PJ132':'fixed-discharge',
                      'PJ133':'fixed-discharge',
                      'PJ134':'fixed-discharge',
                      'PJ135':'fixed-discharge',
                      'PJ136':'fixed-discharge',
                      'PJ145':'variable-discharge',
                      'PJ146':'variable-discharge',
                      'PJ147':'variable-discharge',
                      'PJ148':'variable-discharge',
                      'PJ149':'variable-discharge',
                      'PJ150':'variable-discharge',
                      'PJ151':'variable-discharge',
                      'PJ152':'variable-discharge',
                      }
    """

    for channel in channels:

        cells = cell_map[channel]

        for cell in cells:

            cell_no = int(cell[-3:])
            #cmap = plt.get_cmap(name, 70)
            #cell_ars = []
            dir = '../data/mountgrove/{}/'.format(experiment_map[cell])

            # First get the initial EIS and GCPL
            cycle = 0
            path = path_to_file(channel, cell, cycle, dir)
            ptd = '{}{}/'.format(dir, cell)
            filestart = '{}_{:03d}a_'.format(cell, cycle)

            if cell_no in [145, 146, 147, 148]:
                start_cycle = 3
            else:
                start_cycle = 2

            ptf_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 4, 'GEIS', channel)
            ptf_cv = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 3, 'GCPL', channel)

            # Get features of discharge capacity-voltage curve

            cap0, cvf0, e_out, d_rate, cap_curve_norm = discharge_features(ptf_cv, cycle)

            print('Cell PJ{:03d}\t C0 {:.2f}\t Start cycle {}'.format(cell_no, cap0, start_cycle))
            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, feature_type=feature_type, new_log_freq=new_log_freq, n_repeats=n_repeats)
            eis0 = x_eis

            cell_c1_rates = []
            cell_c2_rates = []
            cell_d_rates = []
            cell_eis_cs = []
            cell_eis_ds = []
            cell_t_charges = []
            cell_t1_charges = []
            cell_t2_charges = []
            cell_ocvs = []
            cell_cap_cs = []
            cell_cap_ds = []
            cell_cycles = []
            cell_cap_nets = []
            cell_cap_throughputs = []
            cell_cvfs = []
            cell_cvfs.append(cvf0)

            cell_last_caps = []
            cell_sohs = []
            cell_last_caps.append(cap0)
            cell_sohs.append(1)

            cap_net = 0
            cap_throughput = 0
            cell_cap_nets.append(cap_net)
            cell_cap_throughputs.append(cap_throughput)



            for cycle in range(start_cycle, start_cycle+30):
                path = path_to_file(channel, cell, cycle, dir)
                ptd = '{}/{}/'.format(dir, cell)
                filestart = '{}_{:03d}_'.format(cell, cycle)

                for step in range(n_steps):
                    true_cycle = 1+(cycle-2)*n_steps + step

                    ptf_discharge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_discharge_no + step*4, 'GEIS', channel)
                    ptf_charge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_charge_no + step*4, 'GCPL', channel)
                    ptf_charge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_charge_no + step*4, 'GEIS', channel)
                    ptf_discharge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_discharge_no + step*4, 'GCPL', channel)

                    ptf_files = [ptf_discharge_eis, ptf_charge_gcpl, ptf_charge_eis, ptf_discharge_gcpl]

                    #check_files(ptf_files)

                    # Get features of discharge EIS spectrum
                    try:
                        eis_d = eis_features(ptf_discharge_eis, feature_type=feature_type, new_log_freq=new_log_freq, n_repeats=1)
                    except:
                        eis_d = None

                    # Compute the time to charge
                    try:
                        t_charge, cap_c, c1_rate, c2_rate, t1_charge, t2_charge, ocv = charge_features(ptf_charge_gcpl)

                    except:
                        t_charge = None
                        cap_c = None
                        c1_rate = None
                        c2_rate = None

                    # Get features of discharge capacity-voltage curve
                    try:
                        cap_d, cvf, _, d_rate, _ = discharge_features(ptf_discharge_gcpl, true_cycle, cap_curve_norm)
                    except:
                        cap_d = None
                        cvf = None
                        d_rate = None

                    if any(elem is None for elem in (eis_d, t_charge, cap_c, cap_d)):
                        #print('{:02d}\t{} {} {}'.format(true_cycle, t_charge, cap_c, cap_d))
                        continue
                    else:
                        cap_net += cap_c - cap_d
                        cap_throughput += cap_c + cap_d
                        cell_cvfs.append(cvf)
                        cell_c1_rates.append(c1_rate)
                        cell_c2_rates.append(c2_rate)
                        cell_d_rates.append(d_rate)
                        cell_eis_ds.append(eis_d)
                        cell_t_charges.append(t_charge)
                        cell_t1_charges.append(t1_charge)
                        cell_t2_charges.append(t2_charge)
                        cell_ocvs.append(ocv)
                        cell_cap_cs.append(cap_c)
                        cell_cap_ds.append(cap_d)
                        cell_sohs.append(cap_d / cap0)
                        cell_last_caps.append(cap_d)
                        cell_cap_nets.append(cap_net)
                        cell_cap_throughputs.append(cap_throughput)
                        cell_cycles.append(1+(cycle-2)*n_steps + step)
                        #print('{:02d}\t{:.1f}\t{:.1f}\t{:.3f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(1+(cycle-2)*n_steps + step, c1_rate, c2_rate, ocv, t_charge, t1, t2, t_charge-t1-t2))

            cell_c1_rates = np.array(cell_c1_rates)
            cell_c2_rates = np.array(cell_c2_rates)
            cell_d_rates = np.array(cell_d_rates)
            cell_eis_ds = np.vstack(np.array(cell_eis_ds))
            cell_cvfs = np.vstack(np.array(cell_cvfs))
            cell_t_charges = np.array(cell_t_charges)
            cell_t1_charges = np.array(cell_t1_charges)
            cell_t2_charges = np.array(cell_t2_charges)
            cell_ocvs = np.array(cell_ocvs)
            cell_cap_cs = np.array(cell_cap_cs)
            cell_cap_ds = np.array(cell_cap_ds)
            cell_cap_nets = np.array(cell_cap_nets)
            cell_cap_throughputs = np.array(cell_cap_throughputs)
            cell_cycles = np.array(cell_cycles)
            cell_last_caps = np.array(cell_last_caps)
            cell_sohs = np.array(cell_sohs)

            if nl == 0:
                cell_idx.append([cell,]*cell_t_charges.shape[0])
                cycles.append(cell_cycles)
                c1_rates.append(cell_c1_rates.reshape(-1, 1))
                c2_rates.append(cell_c2_rates.reshape(-1, 1))
                d_rates.append(cell_d_rates.reshape(-1, 1))
                eis_ds.append(cell_eis_ds)
                t_charges.append(cell_t_charges)
                t1_charges.append(cell_t1_charges)
                t2_charges.append(cell_t2_charges)
                ocvs.append(cell_ocvs)
                cap_cs.append(cell_cap_cs)
                cap_ds.append(cell_cap_ds)
                last_caps.append(cell_last_caps[:-1])
                sohs.append(cell_sohs[:-1])
                cap_nets.append(cell_cap_nets[:-1])
                cap_throughputs.append(cell_cap_throughputs[:-1])
                cvfs.append(cell_cvfs[:-1, :])
                cap_inits.append(cap0*np.ones(cell_t_charges.shape[0]))
                eis_inits.append(np.tile(eis0, (cell_t_charges.shape[0], 1)))
            elif nl == 1:
                cell_idx.append([cell,]*(cell_t_charges.shape[0]-1))
                cycles.append(cell_cycles[:-nl])
                c1_rates.append(np.concatenate((cell_c1_rates[:-nl].reshape(-1, 1), cell_c1_rates[nl:].reshape(-1, 1)), axis=1))
                c2_rates.append(np.concatenate((cell_c2_rates[:-nl].reshape(-1, 1), cell_c2_rates[nl:].reshape(-1, 1)), axis=1))
                d_rates.append(np.concatenate((cell_d_rates[:-nl].reshape(-1, 1), cell_d_rates[nl:].reshape(-1, 1)), axis=1))
                eis_ds.append(cell_eis_ds[:-nl])
                t_charges.append(cell_t_charges)
                t1_charges.append(cell_t1_charges)
                t2_charges.append(cell_t2_charges)
                ocvs.append(cell_ocvs[:-nl])
                cap_cs.append(cell_cap_cs)
                cap_ds.append(cell_cap_ds[nl:])
                cap_nets.append(cell_cap_nets[:-1])
                cap_throughputs.append(cell_cap_throughputs[:-1])
                cvfs.append(cell_cvfs[:-1-nl, :])
                cap_inits.append(cap0*np.ones(cell_t_charges.shape[0]-1))
                eis_inits.append(np.tile(eis0, (cell_t_charges.shape[0]-1, 1)))

    cycles = np.hstack(cycles)
    c1_rates = np.vstack(c1_rates)
    c2_rates = np.vstack(c2_rates)
    d_rates = np.vstack(d_rates)
    eis_ds = np.vstack(eis_ds)
    eis_inits = np.vstack(eis_inits)
    t_charges = np.hstack(t_charges)
    t1_charges = np.hstack(t1_charges)
    t2_charges = np.hstack(t2_charges)
    ocvs = np.hstack(ocvs)
    cap_cs = np.hstack(cap_cs)
    cap_ds = np.hstack(cap_ds)
    last_caps = np.hstack(last_caps)
    sohs = np.hstack(sohs)
    cap_nets = np.hstack(cap_nets)
    cap_throughputs = np.hstack(cap_throughputs)
    cap_inits = np.hstack(cap_inits)
    cell_idx = np.array([item for sublist in cell_idx for item in sublist])
    cvfs = np.vstack(cvfs)
    data = (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates)

    return cell_idx, cap_ds, data


def extract_n_step_data(experiment, channels):

    cell_map = identify_cells(experiment)
    feature_type = 'full'
    n_repeats = 1
    n_steps = 4
    new_log_freq = np.linspace(-1.66, 3.9, 100)

    freq1 = np.log10(2.16)
    freq2 = np.log10(17.8)
    idx_freq1 = np.argmin(np.abs(new_log_freq-freq1))
    idx_freq2 = np.argmin(np.abs(new_log_freq-freq2))

    nl = 0

    states = {}
    actions = {}
    cycles = {}
    cap_ds = {}

    column_map = {
        'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
        'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
    }

    cmap_names = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'RdPu', 'BuPu', 'GnBu', 'YlOrRd']

    experiment_map = {'PJ097':'variable-discharge',
                      'PJ098':'variable-discharge',
                      'PJ099':'variable-discharge',
                      'PJ100':'variable-discharge',
                      'PJ101':'variable-discharge',
                      'PJ102':'variable-discharge',
                      'PJ103':'variable-discharge',
                      'PJ104':'variable-discharge',
                      'PJ105':'variable-discharge',
                      'PJ106':'variable-discharge',
                      'PJ107':'variable-discharge',
                      'PJ108':'variable-discharge',
                      'PJ109':'variable-discharge',
                      'PJ110':'variable-discharge',
                      'PJ111':'variable-discharge',
                      'PJ112':'variable-discharge',
                      'PJ121':'fixed-discharge',
                      'PJ122':'fixed-discharge',
                      'PJ123':'fixed-discharge',
                      'PJ124':'fixed-discharge',
                      'PJ125':'fixed-discharge',
                      'PJ126':'fixed-discharge',
                      'PJ127':'fixed-discharge',
                      'PJ128':'fixed-discharge',
                      'PJ129':'fixed-discharge',
                      'PJ130':'fixed-discharge',
                      'PJ131':'fixed-discharge',
                      'PJ132':'fixed-discharge',
                      'PJ133':'fixed-discharge',
                      'PJ134':'fixed-discharge',
                      'PJ135':'fixed-discharge',
                      'PJ136':'fixed-discharge',
                      'PJ145':'variable-discharge',
                      'PJ146':'variable-discharge',
                      'PJ147':'variable-discharge',
                      'PJ148':'variable-discharge',
                      'PJ149':'variable-discharge',
                      'PJ150':'variable-discharge',
                      'PJ151':'variable-discharge',
                      'PJ152':'variable-discharge',
                      }

    for channel in channels:

        cells = cell_map[channel]

        for cell in cells:

            cell_states = []
            cell_actions = []
            cell_cycles = []
            cell_cap_ds = []

            cell_no = int(cell[-3:])
            #cmap = plt.get_cmap(name, 70)
            #cell_ars = []
            dir = '../data/mountgrove/{}/'.format(experiment_map[cell])

            # First get the initial EIS and GCPL
            cycle = 0
            path = path_to_file(channel, cell, cycle, dir)
            ptd = '{}{}/'.format(dir, cell)
            filestart = '{}_{:03d}a_'.format(cell, cycle)

            ptf_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 4, 'GEIS', channel)
            ptf_cv = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, 3, 'GCPL', channel)

            # Get features of discharge capacity-voltage curve

            cap0, cvf0, e_out, d_rate, cap_curve_norm = discharge_features(ptf_cv, cycle)

            print('Cell PJ{:03d}\t C0 {:.2f}'.format(cell_no, cap0))
            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, feature_type=feature_type, new_log_freq=new_log_freq, n_repeats=n_repeats)
            eis0 = x_eis

            for cycle in range(2, 32):
                path = path_to_file(channel, cell, cycle, dir)
                ptd = '{}/{}/'.format(dir, cell)
                filestart = '{}_{:03d}_'.format(cell, cycle)

                geis_discharge_no = 1
                gcpl_charge_no = 2
                geis_charge_no = 3
                gcpl_discharge_no = 4

                for step in range(n_steps):
                    true_cycle = 1+(cycle-2)*n_steps + step

                    ptf_discharge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_discharge_no + step*4, 'GEIS', channel)
                    ptf_charge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_charge_no + step*4, 'GCPL', channel)
                    ptf_charge_eis = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, geis_charge_no + step*4, 'GEIS', channel)
                    ptf_discharge_gcpl = '{}{}{:02d}_{}_CA{}.txt'.format(ptd, filestart, gcpl_discharge_no + step*4, 'GCPL', channel)

                    ptf_files = [ptf_discharge_eis, ptf_charge_gcpl, ptf_charge_eis, ptf_discharge_gcpl]

                    #check_files(ptf_files)

                    # Get features of discharge EIS spectrum
                    try:
                        eis_d = eis_features(ptf_discharge_eis, feature_type=feature_type, new_log_freq=new_log_freq, n_repeats=1)
                    except:
                        eis_d = None

                    # Compute the time to charge
                    try:
                        t_charge, cap_c, c1_rate, c2_rate, t1_charge, t2_charge, ocv = charge_features(ptf_charge_gcpl)

                    except:
                        t_charge = None
                        cap_c = None
                        c1_rate = None
                        c2_rate = None

                    # Get features of discharge capacity-voltage curve
                    try:
                        cap_d, cvf, _, d_rate, _ = discharge_features(ptf_discharge_gcpl, true_cycle, cap_curve_norm)
                    except:
                        cap_d = None
                        cvf = None
                        d_rate = None

                    if any(elem is None for elem in (eis_d, t_charge, cap_c, cap_d)):
                        #print('{:02d}\t{} {} {}'.format(true_cycle, t_charge, cap_c, cap_d))
                        continue
                    else:
                        cell_states.append(eis_d)
                        cell_actions.append(np.array([d_rate, c1_rate, c2_rate]).reshape(1, -1))
                        cell_cap_ds.append(cap_d)
                        cell_cycles.append(1+(cycle-2)*n_steps + step)
                        #print('{:02d}\t{:.1f}\t{:.1f}\t{:.3f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(1+(cycle-2)*n_steps + step, c1_rate, c2_rate, ocv, t_charge, t1, t2, t_charge-t1-t2))

            cell_states = np.vstack(np.array(cell_states))
            cell_actions = np.vstack(np.array(cell_actions))
            cell_cycles = np.array(cell_cycles)
            cell_cap_ds = np.array(cell_cap_ds)

            states[cell] = cell_states
            actions[cell] = cell_actions
            cycles[cell] = cell_cycles
            cap_ds[cell] = cell_cap_ds

    data = (states, actions, cycles, cap_ds)

    return data

def median_confidence_curve(experiment_name, fnames, experiment='variable-discharge', metric='rmse',
                            figsize=(7, 7), linewidth=3.0, fontsize=24):
    """
    Plot confidence curve.
    :param mean:
    :param variance:
    :param target:
    :param filename:
    :return:
    """

    pts = 'figures/{}/{}_conf_curve_median_{}.png'.format(experiment, experiment_name, metric)

    percentiles = np.arange(100, 4, -5)

    metric_models = []
    metric_oracles = []

    for id_te in range(len(fnames)):

        fname = fnames[id_te]

        mean = np.load('processed/{}/pred_mn_{}_PJ{}.npy'.format(experiment, experiment_name, fname))
        std = np.load('processed/{}/pred_std_{}_PJ{}.npy'.format(experiment, experiment_name, fname))
        target = np.load('processed/{}/true_{}_PJ{}.npy'.format(experiment, experiment_name, fname))

        conf_percentile, metric_model, metric_oracle = metric_ordering(mean, std**2, target, metric)
        indices = []
        for percentile in percentiles:
            indices.append(find_nearest(conf_percentile, percentile))
        indices = np.array(indices)
        metric_models.append(metric_model[indices])
        metric_oracles.append(metric_oracle[indices])

    metric_models = np.array(metric_models)
    metric_oracles = np.array(metric_oracles)
    metric_model_mn = np.mean(metric_models, axis=0)
    metric_model_std = np.std(metric_models, axis=0)
    metric_model_median = np.median(metric_models, axis=0)
    metric_oracle_mn = np.mean(metric_oracles, axis=0)
    metric_oracle_std = np.std(metric_oracles, axis=0)
    metric_oracle_median = np.median(metric_oracles, axis=0)

    mpl.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)

    ax.plot(percentiles, metric_oracle_median, color="C0", linestyle=linestyles['densely dashed'],
            linewidth=linewidth, label="Oracle")
    ax.plot(percentiles, metric_model_median, color="C1", linestyle=linestyles['densely dotted'],
            linewidth=linewidth, label="Model")

    ymin = min(np.min(metric_model_median), np.min(metric_oracle_median))
    ymax = max(np.max(metric_model_median), np.max(metric_oracle_median))

    ax.set_ylim(ymin, ymax)

    #yticks = [0, 4, 8, 12]

    #yticks = np.arange(np.round(ymin, decimals=1), np.round(ymax + 0.2, decimals=1), step=0.2)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(np.round(yticks, decimals=1), fontsize=fontsize)

    xticks = np.linspace(0, 100, 5)
    xticklabels = [0, 25, 50, 75, 100]
    ax.set_xlim(0, 100)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    ax.set_xlabel("Percentage data imputed (%)", fontsize=fontsize)
    if metric == 'rmse':
        ax.set_ylabel("RMSE", fontsize=fontsize)
    elif metric == 'r2':
        ax.set_ylabel("R2 score", fontsize=fontsize)
    elif metric == 'rmae':
        ax.set_ylabel("Relative MAE (%)", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(pts, dpi=400, transparent=True)
    return
