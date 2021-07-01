import ntpath
from multiprocessing import Pool
import os
import time
import readf as readf

from DROTestGround.DRO.readf import read_txt_to_arr3D
import Tissue as t


def read_file(path):
    return t.Tissue(read_txt_to_arr3D(path))


def par_read_file(path):
    start_time = time.time()
    os.chdir(path)
    pool = Pool(processes=os.cpu_count())
    tissue_list = pool.map(read_file, os.listdir(path))
    pool.close()
    pool.join()
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    return tissue_list


def main(path):
    tissue_list = par_read_file(path)
    return tissue_list


if __name__ == '__main__':
    path = "/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/tissues"
    main(path)
