# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math
import ntpath
import Tissue as t

def read_file(dir):
    arr = np.loadtxt(dir,dtype=np.int64)
    size = np.size(arr)
    n = int(round(size ** (1. / 3)))

    if size != math.pow(n, 3):
        raise ValueError("Unable to reshape tissue sample to 3 dimension matrix.")

    arr = arr.reshape(n, n, n)
    return arr


def add_to_list(tissue_list, file_list, path):
    tissue_list.append(t.Tissue(read_file(path)))
    file_list.append(ntpath.basename(path))
