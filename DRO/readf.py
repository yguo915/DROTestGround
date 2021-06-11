# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math



def read_file(dir):
    arr = np.loadtxt(dir)
    size = np.size(arr)
    n = int(round(size ** (1. / 3)))

    if size != math.pow(n, 3):
        raise ValueError("Unable to reshape tissue sample to 3 dimension matrix.")

    arr = arr.reshape(n, n, n)
    return arr
