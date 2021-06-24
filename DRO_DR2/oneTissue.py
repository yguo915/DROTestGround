import os
from itertools import repeat

import numpy as np
import cupy as cp
import time
from multiprocessing import Pool, Process
import scipy.io as sio
from scipy.sparse import csr_matrix, diags
import cupyx.scipy.sparse as csp
from functools import partial
import matplotlib.pyplot as plt
import math
from numba import jit, cuda

import Tissue as t
import Pulse as p
import readf as readf
import concentrationMatrix as cm









def DR2_micro(DR2_micro_arr, signal_arr,time_arr,  tissue, pulse, aiffull, Ktrans, CBF, PP, NR):
    """
    microscopic effect, size of molecule interaction
    :param DR2_micro_arr:
    :param signal_arr:
    :param tissue:
    :param pulse:
    :param time:
    :param aiffull:
    :param Ktrans:
    :param CBF:
    :param PP:
    :param NR:
    :return:
    """
    sus_factor = 2.7e-10
    relaxivity2 = 5.3

    tissue.set_Delta_chi(cm.concentration_matrix(tissue, CBF, Ktrans, aiffull))

    tissue.set_transit_matrix(load_transit_matrix(tissue, pulse))
    tissue.set_relaxation_matrix(set_relaxation_time(tissue, pulse))
    if not check_CFLcondition(tissue, pulse):
        raise ValueError("Wrong CFL condition")

    start_time = time.time()


    # p = Process(target = f, args = (tissue, pulse, DR2_micro_arr, signal_arr, time_arr))
    # p.start()
    # p.join()

    with Pool(processes=os.cpu_count()) as pool:
        DR2_micro_arr[:, NR, PP] = np.array(pool.starmap(get_DR2_micro, zip(repeat(tissue), time_arr)))
        signal_arr[:,NR,PP] = np.array(pool.starmap(get_DSC_signal_cp, zip(repeat(tissue),repeat(pulse), time_arr)))


    tissue.set_DR2_micro(DR2_micro_arr)
    tissue.set_signal(signal_arr)
    M0 = tissue.get_tissue_size()
    R = np.zeros(signal_arr.shape)
    R[:,NR, PP] = - np.log(signal_arr[:,NR,PP]/M0)/pulse.TE
    DR2_meso = R * 10**3
    DR2_total = DR2_meso + DR2_micro_arr

    end_time = time.time()
    total_time = end_time - start_time
    print("run time:", total_time)

    # plt.figure(1)
    # plt.clf()
    # fig, ax = plt.subplots(num=1)
    # ax.plot(np.arange(120), signal_arr.reshape(120, 1), 'k--', label='Python_signal', color="r")
    # ax.legend(loc='best')
    # plt.show()
    #
    plt.figure(1)
    plt.clf()
    fig, ax = plt.subplots(num=1)
    ax.plot(np.arange(120), DR2_total.reshape(120, 1), 'k--', label='Python_DR2_micro', color="b")
    ax.legend(loc='best')
    plt.show()

#
# def f(tissue, pulse, DR2_micro_arr,signal_arr, arr):
#     for i in range (0, len(arr)-1):
#         DR2_micro_arr[i, 0, 0] = get_DR2_micro(tissue,arr, i)
#         signal_arr[i, 0, 0] = get_DSC_signal(tissue, pulse, i)
#



def get_DR2_micro(tissue,t):
    relaxivity2 = 5.3
    return relaxivity2 * (
            tissue.get_Delta_chi()[0, 0, t] + tissue.get_Delta_chi()[3, 0, t])



def get_DSC_signal(tissue, pulse , nfile):
    dephase_matrix = salomir_transformation(tissue,pulse, nfile).reshape(tissue.get_tissue_size())
    transit_matrix = tissue.get_transit_matrix()
    relaxation_matrix = tissue.get_relaxation_matrix()

    M = np.zeros(tissue.get_tissue_size(), dtype = complex)+1

    Nt = int (pulse.TE / pulse.dt)

    if (pulse.type == "ge"):
        for nt in range(0,Nt):
            M = np.multiply(M, relaxation_matrix)
            M = transit_matrix * M
            M = np.multiply(M, dephase_matrix)

            print(nt)

            # for i in range(0,274624):
            #     M[i] = M[i] * dephase_matrix[i]

    #         print(nt)
    sum = np.sum(M)
    return sum


def get_DSC_signal_cp(tissue, pulse ,nfile):
    dephase_matrix = cp.array(salomir_transformation(tissue,pulse, nfile).reshape(tissue.get_tissue_size()))
    transit_matrix = csp.csc_matrix(tissue.get_transit_matrix())
    relaxation_matrix = cp.array(tissue.get_relaxation_matrix())


    M = cp.zeros(tissue.get_tissue_size(), dtype = complex)+1

    Nt = int (pulse.TE / pulse.dt)

    if (pulse.type == "ge"):
        for nt in range(0,Nt):
            M = cp.multiply(M, relaxation_matrix)
            M = transit_matrix * M
            M = cp.multiply(M, dephase_matrix)

            print(nt)

            # for i in range(0,274624):
            #     M[i] = M[i] * dephase_matrix[i]

    #         print(nt)
    M = cp.asnumpy(M)
    sum = np.sum(M)
    return sum

def salomir_transformation(tissue,pulse, nfile):
    tissue_shape = tissue.get_tissue_shape()
    sphere = np.zeros((tissue_shape[0] + 1, tissue_shape[1] + 1, tissue_shape[2] + 1))
    img_susc = np.zeros((tissue_shape[0] + 1, tissue_shape[1] + 1, tissue_shape[2] + 1))
    sphere[0:tissue_shape[0], 0:tissue_shape[1], 0:tissue_shape[2]] = tissue.get_tissue()

    chi_incell = 0
    chi_excell = tissue.get_Delta_chi()[5, 0, nfile]
    chi_vessel = tissue.get_Delta_chi()[2, 0, nfile]

    idx_incell = np.where(sphere == 1)
    idx_excell = np.where(sphere == 0)
    idx_vessel = np.where(sphere == 2)
    img_susc[idx_incell] = chi_incell
    img_susc[idx_excell] = chi_excell
    img_susc[idx_vessel] = chi_vessel
    # img_susc= np.transpose(img_susc, (1,2,0))

    B_vasnpertb = bshift_sal(img_susc, pulse)[int(tissue.pad[0]):tissue_shape[0], int(tissue.pad[1]):tissue_shape[1],
                  int(tissue.pad[2]):tissue_shape[2]]
    DBz = B_vasnpertb.reshape(tissue.get_tissue_size(), 1)

    return np.exp(-1j * pulse.gamma * DBz * pulse.dt)


def bshift_sal(img_susc, pulse):
    img_shape = img_susc.shape
    sz = img_shape[0]
    sx = img_shape[1]
    sy = img_shape[2]

    kx = np.arange(-sx / 2, sx / 2)
    ky = np.arange(-sy / 2, sy / 2)
    kz = np.arange(-sz / 2, sz / 2)

    [kx3D, kz3D, ky3D] = np.array(np.meshgrid(kx, ky, kz))
    kx3D = -1 * kx3D
    ky3D = -1 * ky3D
    kz3D = -1 * kz3D

    k_squared = kx3D ** 2 + ky3D ** 2 + kz3D ** 2
    np.seterr(divide='ignore', invalid='ignore')

    numeratore = 1 / 3 - (kz3D ** 2) / k_squared
    idx = np.where(k_squared == 0)
    numeratore[idx] = 0

    ft_susc_dist = np.fft.fftshift(np.fft.fftn(4 * math.pi * img_susc))
    producto = ft_susc_dist * numeratore
    BSHIFT = - pulse.B0 * (np.fft.ifftn(np.fft.fftshift(producto)))

    return BSHIFT


def set_coord_range(tissue):
    n = tissue.get_tissue_shape()
    [x, y, z] = np.array(np.meshgrid(np.arange(1, n[0] + 1), np.arange(1, n[1] + 1), np.arange(1, n[2] + 1)))
    x = (x - (n[0] + 1) / 2) * tissue.dr[0]
    y = (y - (n[1] + 1) / 2) * tissue.dr[1]
    z = (z - (n[2] + 1) / 2) * tissue.dr[2]

    x_tmp = np.zeros(n)
    y_tmp = np.zeros(n)
    z_tmp = np.zeros(n)

    for i in range(0, n[2]):
        x_tmp[:, :, i] = x[:, :, i]
        y_tmp[:, :, i] = y[:, :, i]
        z_tmp[:, :, i] = z[:, :, i]

    tissue.set_X(x)
    tissue.set_Y(y)
    tissue.set_Z(z)


def check_CFLcondition(tissue, pulse):
    return max(tissue.D * pulse.dt / (tissue.dr ** 2)) < (0.5 / len(tissue.get_tissue_shape()))


def load_transit_matrix(tissue, pulse):
    mat_fname = "C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/transitmatrix_274625.mat"
    mat_contents = sio.loadmat(mat_fname)
    transit_matrix = mat_contents["transitmatrix_274625"]
    jump_probability = tissue.D[0] * pulse.dt / (tissue.dr[0] ** 2)
    uniform_matrix = diags([1], [0], transit_matrix.shape)
    transit_matrix[transit_matrix.nonzero()] = 1 * jump_probability

    return transit_matrix + uniform_matrix * (1 - jump_probability * 7)


def set_relaxation_time(tissue, pulse):
    tissue_arr3D = tissue.get_tissue()
    tissue_arr = tissue_arr3D.reshape(tissue_arr3D.size)
    tissue_arr0 = np.zeros(tissue_arr.shape)
    tissue_arr0 = np.where(tissue_arr == 0, 1, tissue_arr0)
    tissue_arr1 = np.zeros(tissue_arr.shape)
    tissue_arr1 = np.where(tissue_arr == 1,1, tissue_arr1)
    tissue_arr2 = np.zeros(tissue_arr.shape)
    tissue_arr2 = np.where(tissue_arr == 2, 1,tissue_arr2)
    T2star_relaxation = tissue_arr0 * tissue.T2star[0] + tissue_arr1 * tissue.T2star[1] + tissue_arr2 * tissue.T2star[2]
    for i in range(0, T2star_relaxation.size):
        T2star_relaxation[i] = math.exp(-pulse.dt / T2star_relaxation[i])
    return (T2star_relaxation)


def dsc_init(tissue, PP, NR):
    dosing_arr = np.arange(PP + 1)
    rRange_arr = np.arange(NR + 1)
    time_arr = np.concatenate((np.arange(0, 61, 5), np.arange(61, 151, 1), np.linspace(151, 180, 17, dtype=np.int64)))
    signal = np.zeros((time_arr.size, rRange_arr.size, dosing_arr.size))
    DR2_micro_arr = np.zeros((time_arr.size, rRange_arr.size, dosing_arr.size))
    set_coord_range(tissue)

    return dosing_arr, rRange_arr, time_arr, signal, DR2_micro_arr

def main(tissue, arterial_input_function_full, Ktrans, CBF, PP, NR):

    pulse = p.Pulse()
    (dosing_arr, rRange_arr, time_arr, signal_arr, DR2_micro_arr) = dsc_init(tissue, PP, NR)

    # pool = Pool(processes=os.npu_count())
    DR2_micro(DR2_micro_arr, signal_arr,time_arr, tissue, pulse, arterial_input_function_full, Ktrans[NR], CBF[NR], PP,
              NR)



if __name__ == '__main__':
    arterial_input_function_full = np.loadtxt(
        "C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/aiffull_modified.txt")
    Ktrans = np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/Ktrans.txt")
    CBF = np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/CBF.txt")
    Cell_size =np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/CellSize.txt")

    file_path = "C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/tissues/tissueVpVcMets010657_65_65_65.txt"
    os.chdir("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/tissues/")

    PP = 0
    NR = 0
    tissue = t.Tissue(readf.read_txt_to_arr3D(file_path), Cell_size[NR])

    pulse = p.Pulse()
    (dosing_arr, rRange_arr, time_arr, signal_arr, DR2_micro_arr) = dsc_init(PP, NR)

    # pool = Pool(processes=os.npu_count())
    DR2_micro(DR2_micro_arr, signal_arr,time_arr, tissue, pulse, arterial_input_function_full, Ktrans[NR], CBF[NR], PP,
              NR)
    # pool.close()
    # pool.join()
