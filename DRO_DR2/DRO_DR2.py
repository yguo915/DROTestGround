import os
from itertools import repeat

import numpy as np
import time
from multiprocessing import Pool, Process
import scipy.io as sio
from scipy.sparse import csr_matrix, diags
from functools import partial
import matplotlib.pyplot as plt
import math

import Tissue as t
import Pulse as p
import readf as readf
import concentrationMatrix as cm


class DRO_DR2():
    def __init__(self, tissue, pulse, arterial_input_function_full, Ktrans, CBF, transit_matrix, NR, PP):
        self.tissue = tissue
        self.pulse = pulse
        self.arterial_input_function_full = arterial_input_function_full
        self.Ktrans = Ktrans
        self.CBF = CBF
        self.transit_matrix = transit_matrix
        self.NR = NR
        self.PP = PP

        (dosing_arr, rRange_arr, time_arr, signal_arr, DR2_micro_arr) = self.dsc_init()
        self.dosing_arr = dosing_arr
        self.rRange_arr =  rRange_arr
        self.time_arr = time_arr
        self.signal_arr = signal_arr
        self.DR2_micro_arr = DR2_micro_arr

        self.tissue.set_Delta_chi(cm.concentration_matrix(tissue, CBF, Ktrans, arterial_input_function_full))
        self.tissue.set_transit_matrix(self.load_transit_matrix)
        self.tissue.set_relaxation_matrix(self.set_relaxation_time())
        if not self.check_CFLcondition():
            raise ValueError("Wrong CFL condition")


    def get_DR2_total(self):
        sus_factor = 2.7e-10
        relaxivity2 = 5.3

        start_time = time.time()

        with Pool(processes=os.cpu_count()) as pool:
            self.DR2_micro_arr[:, self.NR, self.PP] = np.array(pool.starmap(self.get_DR2_micro, zip(repeat(self.tissue), self.time_arr)))
            self.signal_arr[:, self.NR, self.PP] = np.array(
                pool.starmap(get_DSC_signal, zip(repeat(self.tissue), repeat(self.pulse), self.time_arr)))

        self.tissue.set_DR2_micro(self.DR2_micro_arr)
        self.tissue.set_signal(self.signal_arr)
        M0 = self.tissue.get_tissue_size()
        R = np.zeros(self.signal_arr.shape)
        R[:, self.NR, self.PP] = - np.log(self.signal_arr[:, self.NR, self.PP] / M0) / self.pulse.TE
        DR2_meso = R * 10 ** 3
        DR2_total = DR2_meso + self.DR2_micro_arr

        end_time = time.time()
        total_time = end_time - start_time
        print("run time:", total_time)

        plt.figure(1)
        plt.clf()
        fig, ax = plt.subplots(num=1)
        ax.plot(np.arange(120), DR2_total.reshape(120, 1), 'k--', label='Python_DR2_micro', color="b")
        ax.legend(loc='best')
        plt.show()

    def get_DR2_micro(self, t):
        relaxivity2 = 5.3
        return relaxivity2 * (
                self.tissue.get_Delta_chi()[0, 0, t] + self.tissue.get_Delta_chi()[3, 0, t])

    def load_transit_matrix(self):
        jump_probability = self.tissue.D[0] * self.pulse.dt / (self.tissue.dr[0] ** 2)
        uniform_matrix = diags([1], [0], self.transit_matrix.shape)
        self.transit_matrix[self.transit_matrix.nonzero()] = 1 * jump_probability

        return self.transit_matrix + uniform_matrix * (1 - jump_probability * 7)

    def dsc_init(self):
        dosing_arr = np.arange(self.PP + 1)
        rRange_arr = np.arange(self.NR + 1)
        time_arr = np.concatenate(
            (np.arange(0, 61, 5), np.arange(61, 151, 1), np.linspace(151, 180, 17, dtype=np.int64)))
        signal = np.zeros((time_arr.size, rRange_arr.size, dosing_arr.size))
        DR2_micro_arr = np.zeros((time_arr.size, rRange_arr.size, dosing_arr.size))
        self.set_coord_range()

        return dosing_arr, rRange_arr, time_arr, signal, DR2_micro_arr

    def set_coord_range(self):
        n = self.tissue.get_tissue_shape()
        [x, y, z] = np.meshgrid(np.arange(1, n[0] + 1), np.arange(1, n[1] + 1), np.arange(1, n[2] + 1))
        x = (x - (n[0] + 1) / 2) * self.tissue.dr[0]
        y = (y - (n[1] + 1) / 2) * self.tissue.dr[1]
        z = (z - (n[2] + 1) / 2) * self.tissue.dr[2]

        x_tmp = np.zeros(n)
        y_tmp = np.zeros(n)
        z_tmp = np.zeros(n)

        for i in range(0, n[2]):
            x_tmp[:, :, i] = x[:, :, i]
            y_tmp[:, :, i] = y[:, :, i]
            z_tmp[:, :, i] = z[:, :, i]

        self.tissue.set_X(x)
        self.tissue.set_Y(y)
        self.tissue.set_Z(z)

    def set_relaxation_time(self):
        tissue_arr3D = self.tissue.get_tissue()
        tissue_arr = tissue_arr3D.reshape(tissue_arr3D.size)
        tissue_arr0 = np.zeros(tissue_arr.shape)
        tissue_arr0 = np.where(tissue_arr == 0, 1, tissue_arr0)
        tissue_arr1 = np.zeros(tissue_arr.shape)
        tissue_arr1 = np.where(tissue_arr == 1, 1, tissue_arr1)
        tissue_arr2 = np.zeros(tissue_arr.shape)
        tissue_arr2 = np.where(tissue_arr == 2, 1, tissue_arr2)
        T2star_relaxation = tissue_arr0 * self.tissue.T2star[0] + tissue_arr1 * self.tissue.T2star[1] + tissue_arr2 * \
                            self.tissue.T2star[2]
        for i in range(0, T2star_relaxation.size):
            T2star_relaxation[i] = math.exp(-self.pulse.dt / T2star_relaxation[i])
        return (T2star_relaxation)

    def check_CFLcondition(self):
        return max(self.tissue.D * self.pulse.dt / (self.tissue.dr ** 2)) < (0.5 / len(self.tissue.get_tissue_shape()))
