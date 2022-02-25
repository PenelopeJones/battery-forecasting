# First look at and plot the EIS spectra of the cells over the course of their cycle life to now
import os
import sys
sys.path.append('../')
import pickle
import time

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import simps
from scipy.stats import iqr
import numpy as np
import pandas as pd

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
                  'PJ200':'15-minutes',
                  'PJ201':'15-minutes',
                  'PJ202':'15-minutes',
                  'PJ203':'15-minutes',
                  'PJ204':'15-minutes',
                  'PJ205':'15-minutes',
                  'PJ206':'15-minutes',
                  'PJ207':'15-minutes',
                  'PJ208':'15-minutes',
                  'PJ209':'15-minutes',
                  'PJ210':'15-minutes',
                  'PJ211':'15-minutes',
                  'PJ212':'15-minutes',
                  'PJ213':'15-minutes',
                  'PJ214':'15-minutes',
                  'PJ215':'15-minutes',
                  'PJ216':'15-minutes',
                  'PJ217':'15-minutes',
                  'PJ218':'15-minutes',
                  'PJ219':'15-minutes',
                  'PJ220':'15-minutes',
                  'PJ221':'15-minutes',
                  'PJ222':'15-minutes',
                  'PJ223':'15-minutes',
                  'PJ224':'15-minutes',
                  'PJ225':'15-minutes',
                  'PJ226':'15-minutes',
                  'PJ227':'15-minutes',
                  'PJ228':'15-minutes',
                  'PJ229':'15-minutes',
                  'PJ230':'15-minutes',
                  'PJ231':'15-minutes',
                  'PJ247':'variable-discharge-type2',
                  'PJ248':'variable-discharge-type2',
                  'PJ249':'variable-discharge-type2',
                  'PJ250':'variable-discharge-type2',
                  'PJ251':'variable-discharge-type2',
                  'PJ252':'variable-discharge-type2',
                  'PJ253':'variable-discharge-type2',
                  'PJ254':'variable-discharge-type2',
                  'PJ255':'variable-discharge-type2',
                  'PJ256':'variable-discharge-type2',
                  'PJ257':'variable-discharge-type2',
                  'PJ258':'variable-discharge-type2',
                  'PJ259':'variable-discharge-type2',
                  'PJ260':'variable-discharge-type2',
                  'PJ261':'variable-discharge-type2',
                  'PJ262':'variable-discharge-type2',
                  'PJ263':'variable-discharge-type2',
                  'PJ264':'variable-discharge-type2',
                  'PJ265':'variable-discharge-type2',
                  'PJ266':'variable-discharge-type2',
                  'PJ267':'variable-discharge-type2',
                  'PJ268':'variable-discharge-type2',
                  'PJ269':'variable-discharge-type2',
                  'PJ270':'variable-discharge-type2',
                  'PJ271':'variable-discharge-type2',
                  'PJ272':'variable-discharge-type2',
                  'PJ273':'variable-discharge-type2',
                  'PJ274':'variable-discharge-type2',
                  'PJ275':'variable-discharge-type2',
                  'PJ276':'variable-discharge-type2',
                  'PJ277':'variable-discharge-type2',
                  'PJ278':'variable-discharge-type2',
                  'PJ279':'variable-discharge-type2',
                  }
column_map = {
    'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
    'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
}

cmap_names = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'RdPu', 'BuPu', 'GnBu', 'YlOrRd']



def extract_data_type2(experiment, channels, suffix='vd2'):
    """
    Code to extract data for the variable discharge data (type 2 cells).
    """
    cell_map = identify_cells(experiment)
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
    v_maxs = []

    column_map = {
        'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
        'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
    }

    geis_discharge_no = 1
    gcpl_charge_no = 2
    gcpl_discharge_no = 4

    for channel in channels:

        cells = cell_map[channel]
        print(cells)

        for cell in cells:

            cell_no = int(cell[-3:])
            #cmap = plt.get_cmap(name, 70)
            #cell_ars = []
            dir = '../data/raw-data/{}/'.format(experiment_map[cell])

            # First get the initial EIS and GCPL
            cycle = 0
            path = path_to_file(channel, cell, cycle, dir)
            ptd = '{}{}/'.format(dir, cell)
            filestart = '{}_{:03d}_'.format(cell, cycle)

            ptf_eis = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, 3, 'GEIS', channel)
            ptf_cv = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, 2, 'GCPL', channel)

            # Get features of discharge capacity-voltage curve

            cap0, cvf0, e_out, d_rate, cap_curve_norm = discharge_features(ptf_cv, cycle)

            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, new_log_freq=new_log_freq, n_repeats=n_repeats)
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

            if cell_no in [202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215]:
                start_cycle = 2
            else:
                start_cycle = 2

            if cell_no in [204, 205, 206, 207, 212, 213, 214, 215]:
                v_max = 4.2
            else:
                v_max = 4.3

            print('Cell PJ{:03d}\t C0 {:.2f}\t Start cycle {}'.format(cell_no, cap0, start_cycle))
            for cycle in range(start_cycle, start_cycle+25):
                path = path_to_file(channel, cell, cycle, dir)
                ptd = '{}/{}/'.format(dir, cell)
                filestart = '{}_{:03d}_'.format(cell, cycle)

                for step in range(n_steps):
                    true_cycle = 1+(cycle-2)*n_steps + step

                    ptf_discharge_eis = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, geis_discharge_no + step*4, 'GEIS', channel)
                    ptf_charge_gcpl = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, gcpl_charge_no + step*4, 'GCPL', channel)
                    #ptf_charge_eis = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, geis_charge_no + step*3, 'GEIS', channel)
                    ptf_discharge_gcpl = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, gcpl_discharge_no + step*4, 'GCPL', channel)

                    ptf_files = [ptf_discharge_eis, ptf_charge_gcpl, ptf_discharge_gcpl]

                    check_files(ptf_files)

                    # Get features of discharge EIS spectrum
                    try:
                        eis_d = eis_features(ptf_discharge_eis, new_log_freq=new_log_freq, n_repeats=1)
                    except:
                        eis_d = None

                    # Compute the time to charge
                    try:
                        t_charge, cap_c, c1_rate, c2_rate, ocv = charge_features(ptf_charge_gcpl, suffix=suffix)


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
            cell_ocvs = np.array(cell_ocvs)
            cell_cap_cs = np.array(cell_cap_cs)
            cell_cap_ds = np.array(cell_cap_ds)
            cell_cap_nets = np.array(cell_cap_nets)
            cell_cap_throughputs = np.array(cell_cap_throughputs)
            cell_cycles = np.array(cell_cycles)
            cell_last_caps = np.array(cell_last_caps)
            cell_sohs = np.array(cell_sohs)

            cell_idx.append([cell,]*cell_t_charges.shape[0])
            v_maxs.append(v_max*np.ones(cell_t_charges.shape[0]))
            cycles.append(cell_cycles)
            c1_rates.append(cell_c1_rates.reshape(-1, 1))
            c2_rates.append(cell_c2_rates.reshape(-1, 1))
            d_rates.append(cell_d_rates.reshape(-1, 1))
            eis_ds.append(cell_eis_ds)
            t_charges.append(cell_t_charges)
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

    cycles = np.hstack(cycles)
    c1_rates = np.vstack(c1_rates)
    c2_rates = np.vstack(c2_rates)
    d_rates = np.vstack(d_rates)
    eis_ds = np.vstack(eis_ds)

    eis_inits = np.vstack(eis_inits)
    t_charges = np.hstack(t_charges)
    ocvs = np.hstack(ocvs)
    cap_cs = np.hstack(cap_cs)
    cap_ds = np.hstack(cap_ds)
    last_caps = np.hstack(last_caps)
    sohs = np.hstack(sohs)
    cap_nets = np.hstack(cap_nets)
    cap_throughputs = np.hstack(cap_throughputs)
    cap_inits = np.hstack(cap_inits)
    v_maxs = np.hstack(v_maxs)
    cell_idx = np.array([item for sublist in cell_idx for item in sublist])
    cvfs = np.vstack(cvfs)
    data = (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates, v_maxs)
    np.save('cell_{}.npy'.format(suffix), cell_idx)
    np.save('cap_ds_{}.npy'.format(suffix), cap_ds)
    np.save('v_maxs_{}.npy'.format(suffix), v_maxs)
    np.save('last_caps_{}.npy'.format(suffix), last_caps)
    np.save('soh_{}.npy'.format(suffix), sohs)
    np.save('eis_{}.npy'.format(suffix), eis_ds)
    np.save('eis_inits_{}.npy'.format(suffix), eis_inits)
    np.save('cvfs_{}.npy'.format(suffix), cvfs)
    np.save('ocvs_{}.npy'.format(suffix), ocvs)
    np.save('cap_throughputs_{}.npy'.format(suffix), cap_throughputs)
    np.save('d_rates_{}.npy'.format(suffix), d_rates)
    np.save('c1_rates_{}.npy'.format(suffix), c1_rates)
    np.save('c2_rates_{}.npy'.format(suffix), c2_rates)
    print('Saved data')

    return cell_idx, cap_ds, data


def extract_data(experiment, channels):
    """
    Code to extract data for the 15-minutes charging data
    """
    cell_map = identify_cells(experiment)
    n_repeats = 1
    n_steps = 5
    new_log_freq = np.linspace(-1.66, 3.9, 100)

    freq1 = np.log10(2.16)
    freq2 = np.log10(17.8)
    idx_freq1 = np.argmin(np.abs(new_log_freq-freq1))
    idx_freq2 = np.argmin(np.abs(new_log_freq-freq2))

    nl = 0

    c1_rates = []
    c2_rates = []
    d_rates = []
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
    v_maxs = []

    column_map = {
        'GCPL': ['time', 'ewe', 'i', 'capacity', 'power', 'ox/red', 'unnamed'],
        'GEIS': ['freq', 're_z', '-im_z', 'time', 'unnamed']
    }

    geis_discharge_no = 1
    gcpl_charge_no = 2
    gcpl_discharge_no = 3

    for channel in channels:

        cells = cell_map[channel]
        print(cells)

        for cell in cells:

            cell_no = int(cell[-3:])
            #cmap = plt.get_cmap(name, 70)
            #cell_ars = []
            dir = '../data/raw-data/{}/'.format(experiment_map[cell])

            # First get the initial EIS and GCPL
            cycle = 0
            path = path_to_file(channel, cell, cycle, dir)
            ptd = '{}{}/'.format(dir, cell)
            filestart = '{}_{:03d}_'.format(cell, cycle)

            ptf_eis = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, 3, 'GEIS', channel)
            ptf_cv = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, 2, 'GCPL', channel)

            # Get features of discharge capacity-voltage curve

            cap0, cvf0, e_out, d_rate, cap_curve_norm = discharge_features(ptf_cv, cycle)

            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, new_log_freq=new_log_freq, n_repeats=n_repeats)
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

            if cell_no in [202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215]:
                start_cycle = 2
            else:
                start_cycle = 1

            if cell_no in [204, 205, 206, 207, 212, 213, 214, 215]:
                v_max = 4.2
            else:
                v_max = 4.3

            print('Cell PJ{:03d}\t C0 {:.2f}\t Start cycle {}'.format(cell_no, cap0, start_cycle))
            for cycle in range(start_cycle, start_cycle+25):
                path = path_to_file(channel, cell, cycle, dir)
                ptd = '{}/{}/'.format(dir, cell)
                filestart = '{}_{:03d}_'.format(cell, cycle)

                for step in range(n_steps):
                    true_cycle = 1+(cycle-2)*n_steps + step

                    ptf_discharge_eis = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, geis_discharge_no + step*3, 'GEIS', channel)
                    ptf_charge_gcpl = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, gcpl_charge_no + step*3, 'GCPL4', channel)
                    #ptf_charge_eis = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, geis_charge_no + step*3, 'GEIS', channel)
                    ptf_discharge_gcpl = '{}{}{:02d}_{}_C{}.txt'.format(ptd, filestart, gcpl_discharge_no + step*3, 'GCPL', channel)

                    ptf_files = [ptf_discharge_eis, ptf_charge_gcpl, ptf_discharge_gcpl]

                    check_files(ptf_files)

                    # Get features of discharge EIS spectrum
                    try:
                        eis_d = eis_features(ptf_discharge_eis, new_log_freq=new_log_freq, n_repeats=1)
                    except:
                        eis_d = None

                    # Compute the time to charge
                    try:
                        t_charge, cap_c, c1_rate, c2_rate, ocv = charge_features(ptf_charge_gcpl)


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
            cell_ocvs = np.array(cell_ocvs)
            cell_cap_cs = np.array(cell_cap_cs)
            cell_cap_ds = np.array(cell_cap_ds)
            cell_cap_nets = np.array(cell_cap_nets)
            cell_cap_throughputs = np.array(cell_cap_throughputs)
            cell_cycles = np.array(cell_cycles)
            cell_last_caps = np.array(cell_last_caps)
            cell_sohs = np.array(cell_sohs)

            cell_idx.append([cell,]*cell_t_charges.shape[0])
            v_maxs.append(v_max*np.ones(cell_t_charges.shape[0]))
            cycles.append(cell_cycles)
            c1_rates.append(cell_c1_rates.reshape(-1, 1))
            c2_rates.append(cell_c2_rates.reshape(-1, 1))
            d_rates.append(cell_d_rates.reshape(-1, 1))
            eis_ds.append(cell_eis_ds)
            t_charges.append(cell_t_charges)
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

    cycles = np.hstack(cycles)
    c1_rates = np.vstack(c1_rates)
    c2_rates = np.vstack(c2_rates)
    d_rates = np.vstack(d_rates)
    eis_ds = np.vstack(eis_ds)

    eis_inits = np.vstack(eis_inits)
    t_charges = np.hstack(t_charges)
    ocvs = np.hstack(ocvs)
    cap_cs = np.hstack(cap_cs)
    cap_ds = np.hstack(cap_ds)
    last_caps = np.hstack(last_caps)
    sohs = np.hstack(sohs)
    cap_nets = np.hstack(cap_nets)
    cap_throughputs = np.hstack(cap_throughputs)
    cap_inits = np.hstack(cap_inits)
    v_maxs = np.hstack(v_maxs)
    cell_idx = np.array([item for sublist in cell_idx for item in sublist])
    cvfs = np.vstack(cvfs)
    data = (last_caps, sohs, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates, v_maxs)
    np.save('cell_15m.npy', cell_idx)
    np.save('cap_ds_15m.npy', cap_ds)
    np.save('v_maxs_15m.npy', v_maxs)
    np.save('last_caps_15m.npy', last_caps)
    np.save('soh_15m.npy', sohs)
    np.save('eis_15m.npy', eis_ds)
    np.save('eis_inits_15m.npy', eis_inits)
    np.save('cvfs_15m.npy', cvfs)
    np.save('ocvs_15m.npy', ocvs)
    np.save('cap_throughputs_15m.npy', cap_throughputs)
    np.save('d_rates_15m.npy', d_rates)
    np.save('c1_rates_15m.npy', c1_rates)
    np.save('c2_rates_15m.npy', c2_rates)
    print('Saved data')

    return cell_idx, cap_ds, data

# Identify the filenames associated with particular cell cycle
def path_to_file(channel, cell, cycle, dir='data/main/'):
    sub_dir = '{}{}_{}/'.format(dir, cell, channel)
    file_start = '{}_{:03d}_'.format(cell, cycle)
    return sub_dir + file_start

def check_files(ptf_files):
    for ptf in ptf_files:
        if not os.path.isfile(ptf):
            print(ptf)
        else:
            continue
    return

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

def general_sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(b*(x-c)))

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

def eis_to_ecm(eis, new_log_freq, feature_type='randles'):
    n_features = eis.shape[1] // 2
    re_z = eis[:, :n_features]
    im_z = eis[:, n_features:]
    assert re_z.shape == im_z.shape

    if feature_type == 'extended-randles':
        x = np.zeros((re_z.shape[0], 6))
        for i in range(re_z.shape[0]):
            ls = least_squares(overall_fitness_er, x0=(1, 0.1, 0.1, 1, 1, 0.3), bounds=([0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 100, 100]), args=(10**new_log_freq, re_z[i, :], im_z[i, :]))
            x[i, :] = ls.x.reshape(-1)

    elif feature_type == 'randles':
        x = np.zeros((re_z.shape[0], 4))
        for i in range(re_z.shape[0]):
            ls = least_squares(overall_fitness_r, x0=(1, 0.1, 0.1, 0.3), bounds=([0, 0, 0, 0], [100, 100, 100, 100]), args=(10**new_log_freq, re_z[i, :], im_z[i, :]))
            x[i, :] = ls.x.reshape(-1)
    return x

def eis_features(path, new_log_freq=np.linspace(-1.66, 3.9, 500), n_repeats=1):
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

    return np.hstack((re_z, im_z)).reshape(-1)

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

def charge_features(ptf, suffix='15m'):
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
            ocv = np.mean(df.iloc[0:1].ewe.to_numpy())
        except:
            ocv = None

        df_charge = df.loc[df.i > 0.0]
        df_charge = df_charge.reset_index(drop=True)


        charge_cap = df_charge.capacity.max()
        voltage_max = df_charge.ewe.max()

        #print('{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(cap1, cap2, computed_cap, charge_cap))

        if suffix == '15m':
            c_rate = np.mean(df_charge.iloc[5:10].i.to_numpy())
            t_charge = df_charge.time.max() - df_charge.time.min()
            return t_charge / 3600, charge_cap, c_rate, charge_cap*3600/t_charge, ocv
        else:
            idx = df.loc[df.i == 0.0].index.to_numpy()
            change = np.array(np.where((idx[1:] - idx[:-1]) != 1)).reshape(-1)
            assert change.shape[0] == 2, "Error in computing time to change"
            idx_start1 = idx[change[0]]
            idx_change1 = idx[change[0]+1]
            idx_start2 = idx[change[1]]
            idx_change2 = idx[change[1]+1]

            df1_charge = df.loc[(df.i > 0.0) & (df.time < df.iloc[idx_change1].time)]
            df2_charge = df.loc[(df.i > 0.0) & (df.time > df.iloc[idx_start2].time)]

            t1 = df1_charge.time.max() - df1_charge.time.min()
            t2 = df2_charge.time.max() - df2_charge.time.min()

            c1_rate = np.mean(df1_charge.i.to_numpy())
            c2_rate = np.mean(df2_charge.i.to_numpy())

            cap1 = df1_charge.capacity.max()
            cap2 = df2_charge.capacity.max()

            t_charge = df_charge.time.max() - df_charge.time.min() - (df.iloc[idx_start2].time - df.iloc[idx_change1].time)

            return t_charge / 3600, charge_cap, c1_rate, c2_rate, ocv

def extract_input(input_name, data=None, suffix='15m'):
    if data is not None:
        (c, soh, eis_ds, cvfs, ocvs, cap_throughputs, d_rates, c1_rates, c2_rates, v_maxs) = data
    else:
        c = np.load('last_caps_{}.npy'.format(suffix))
        soh = np.load('soh_{}.npy'.format(suffix))
        eis_ds = np.load('eis_{}.npy'.format(suffix))
        cvfs = np.load('cvfs_{}.npy'.format(suffix))
        ocvs = np.load('ocvs_{}.npy'.format(suffix))
        cap_throughputs = np.load('cap_throughputs_{}.npy'.format(suffix))
        d_rates = np.load('d_rates_{}.npy'.format(suffix))
        c1_rates = np.load('c1_rates_{}.npy'.format(suffix))
        c2_rates = np.load('c2_rates_{}.npy'.format(suffix))
        v_maxs = np.load('v_maxs_{}.npy'.format(suffix))

    if suffix == '15m':
        actions = np.hstack([v_maxs.reshape(-1, 1), c1_rates.reshape(-1, 1)])
    elif suffix == 'vd2':
        actions = np.hstack([c1_rates.reshape(-1, 1), c2_rates.reshape(-1, 1), d_rates.reshape(-1, 1)])
    #actions = c1_rates.reshape(-1, 1)

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

    elif input_name == 'ecmr-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='randles')
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmr-cvfs-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='randles')
        states = np.concatenate((states, cvfs), axis=1)
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmer-cvfs-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='extended-randles')
        states = np.concatenate((states, cvfs), axis=1)
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmer-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='extended-randles')
        print(states.shape)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmr-cvfs-ct-c-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='randles')
        states = np.concatenate((states, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        states = np.concatenate((states, c.reshape(-1, 1)), axis=1)
        x = np.concatenate((states, actions), axis=1)

    elif input_name == 'ecmer-cvfs-ct-c-actions':
        states = eis_to_ecm(eis_ds, new_log_freq=np.linspace(-1.66, 3.9, 100), feature_type='extended-randles')
        states = np.concatenate((states, cvfs), axis=1)
        states = np.concatenate((states, cap_throughputs.reshape(-1, 1)), axis=1)
        states = np.concatenate((states, c.reshape(-1, 1)), axis=1)
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

def identify_cells(experiment):
    if experiment == 'variable-discharge':
        cell_map = {'A1':['PJ097','PJ105','PJ145'],
                     'A2':['PJ098','PJ106','PJ146'],
                     'A3':['PJ099','PJ107','PJ147'],
                     'A4':['PJ100','PJ108','PJ148'],
                     'A5':['PJ101','PJ109','PJ149'],
                     'A6':['PJ102','PJ110','PJ150'],
                     'A7':['PJ103','PJ111','PJ151'],
                     'A8':['PJ104','PJ112','PJ152']}

    elif experiment == 'fixed-discharge':
        cell_map = {'A1':['PJ121', 'PJ129',],
                'A2':['PJ122', 'PJ130',],
                'A3':['PJ125', 'PJ131',],
                'A4':['PJ126', 'PJ132',],
                'A5':['PJ123', 'PJ133',],
                'A6':['PJ124', 'PJ134',],
                'A7':['PJ127', 'PJ135',],
                'A8':['PJ128', 'PJ136',],}
    elif experiment == '15-minutes':
        cell_map = {'A1':['PJ200','PJ216'],
                'A2':['PJ201','PJ217'],
                'A3':['PJ202','PJ218'],
                'A4':['PJ203','PJ219'],
                'A5':['PJ204','PJ220'],
                'A6':['PJ205','PJ221'],
                'A7':['PJ206','PJ222'],
                'A8':['PJ207','PJ223'],
                'B1':['PJ208','PJ224'],
                'B2':['PJ209','PJ225'],
                'B3':['PJ210','PJ226'],
                'B4':['PJ211','PJ227'],
                'B5':['PJ212','PJ228'],
                'B6':['PJ213','PJ229'],
                'B7':['PJ214','PJ230'],
                'B8':['PJ215','PJ231'],}
    elif experiment == 'variable-discharge-type2':
        cell_map = {'A1':['PJ248','PJ264',],
                'A2':['PJ249','PJ265',],
                'A3':['PJ250','PJ266',],
                'A4':['PJ251','PJ267',],
                'A5':['PJ252','PJ268',],
                'A6':['PJ253','PJ269',],
                'A7':['PJ254','PJ270',],
                'A8':['PJ255','PJ271',],
                'B1':['PJ256','PJ272',],
                'B2':['PJ257','PJ273',],
                'B3':['PJ258','PJ274',],
                'B4':['PJ259','PJ275',],
                'B5':['PJ260','PJ276',],
                'B6':['PJ261','PJ277',],
                'B7':['PJ262','PJ278',],
                'B8':['PJ263','PJ279',],}

    elif experiment == 'both':
        cell_map = {'A1':['PJ097','PJ105', 'PJ121', 'PJ129', 'PJ145'],
                'A2':['PJ098','PJ106', 'PJ122', 'PJ130', 'PJ146'],
                'A3':['PJ099','PJ107', 'PJ125', 'PJ131', 'PJ147'],
                'A4':['PJ100','PJ108','PJ126', 'PJ132', 'PJ148'],
                'A5':['PJ101','PJ109','PJ123', 'PJ133', 'PJ149'],
                'A6':['PJ102','PJ110','PJ124', 'PJ134', 'PJ150'],
                'A7':['PJ103','PJ111','PJ127', 'PJ135', 'PJ151'],
                'A8':['PJ104','PJ112','PJ128', 'PJ136', 'PJ152'],}
    elif experiment == 'both-variable':
        cell_map = {'A1':['PJ097','PJ105', 'PJ121', 'PJ129', 'PJ145'],
                'A2':['PJ098','PJ106', 'PJ122', 'PJ130', 'PJ146'],
                'A3':['PJ099','PJ107', 'PJ125', 'PJ131', 'PJ147'],
                'A4':['PJ100','PJ108','PJ126', 'PJ132', 'PJ148'],
                'A5':['PJ101','PJ109','PJ123', 'PJ133', 'PJ149'],
                'A6':['PJ102','PJ110','PJ124', 'PJ134', 'PJ150'],
                'A7':['PJ103','PJ111','PJ127', 'PJ135', 'PJ151'],
                'A8':['PJ104','PJ112','PJ128', 'PJ136', 'PJ152'],}

    else:
        cell_map = None

    return cell_map

def extract_n_step_data(experiment, channels):

    cell_map = identify_cells(experiment)
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
            dir = '../data/{}/'.format(experiment_map[cell])

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

            print('Cell PJ{:03d}\t C0 {:.2f}'.format(cell_no, cap0))
            oldcap = e_out

            # Get initial features of discharge EIS spectrum
            x_eis = eis_features(ptf_eis, new_log_freq=new_log_freq, n_repeats=n_repeats)
            eis0 = x_eis

            for cycle in range(start_cycle, start_cycle+30):
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
                        eis_d = eis_features(ptf_discharge_eis, new_log_freq=new_log_freq, n_repeats=1)
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

def ensemble_predict(x, experiment, input_name, n_ensembles=10):
    y_preds = []
    dts = '../results/{}'.format(experiment)
    for i in range(n_ensembles):
        experiment_name = '{}_n1_xgb'.format(input_name)
        with open('{}/models/{}_{}.pkl'.format(dts, experiment_name, i), 'rb') as f:
            regr = pickle.load(f)
            y_pred = regr.predict(x)
            y_preds.append(y_pred.reshape(1, y_pred.shape[0], -1))
    y_preds = np.vstack(y_preds)
    y_pred = np.mean(y_preds, axis=0)
    y_pred_err = np.sqrt(np.var(y_preds, axis=0))
    return y_pred.reshape(-1), y_pred_err.reshape(-1)
