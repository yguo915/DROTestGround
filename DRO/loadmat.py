from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.sparse import csr_matrix, csc
import numpy as np

def load_mat():
    # data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    # mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')
    mat_fname = "/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/input/transitmatrix_274625.mat"
    mat_contents = sio.loadmat(mat_fname)
    transit_matrix = mat_contents["transitmatrix_274625"]
    print(type(transit_matrix))
    print(transit_matrix[0])


if __name__ == '__main__':
    load_mat()