import numpy as np
import cupy as cp
import math
import ntpath
import Tissue as t


def read_txt_to_arr3D(path):
    if not (path.endswith(".txt")):
        raise OSError("Wrong input file type")
    arr = np.loadtxt(path, dtype=np.int64)
    size = np.size(arr)
    n = int(round(size ** (1. / 3)))

    if size != math.pow(n, 3):
        raise ValueError("Unable to reshape tissue sample to 3 dimension matrix.")

    arr = arr.reshape(n, n, n)
    return arr


def read_to_dict(file_tissue_dict, path):
    file_tissue_dict[ntpath.basename(path)] = t.Tissue(read_txt_to_arr3D(path))
