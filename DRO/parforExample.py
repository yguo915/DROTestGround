from multiprocessing import Pool, cpu_count

try:
    import cPickle as pickle
except:
    import pickle
import itertools as it

import numpy as np


def func(t,s):
    s = 1
    for i in range(5):
        s = s + i
    print(t, s)


def parfor1(n):
    pool = Pool(processes=cpu_count())
    pool.map(func, range(n))
    pool.close()
    pool.join()
    print("done")


if __name__ == '__main__':
    parfor1(10)
