# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math
import os


def readf(dir):
    if not os.path.exists(dir):
        raise OSError("File not found.")

    arr = np.loadtxt(dir)
    size = np.size(arr)
    n = int(round(size ** (1. / 3)))

    if size != math.pow(n, 3):
        raise ValueError("Wrong tissue data. Unable to reshape.")

    arr = arr.reshape(n, n, n)
    return arr
