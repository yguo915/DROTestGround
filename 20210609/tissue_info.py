import numpy as np

def get_zeros(tissue_arr3D):
    return np.count_nonzero(tissue_arr3D == 0)


def get_ones(tissue_arr3D):
    return np.count_nonzero(tissue_arr3D == 1)


def get_twos(tissue_arr3D):
    return np.count_nonzero(tissue_arr3D == 2)

