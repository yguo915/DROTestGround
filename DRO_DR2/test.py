import tkinter as tk
from tkinter.filedialog import askopenfilename

import numpy as np
import cupy as cp
import os
import scipy.io as sio

import Tissue as t
import Pulse as p
import readf as readf
import DR2 as dr2
import concentrationMatrix as cm


arterial_input_function_full = np.array(np.loadtxt(
            "C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/aiffull_modified.txt"))
Ktrans = np.array(np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/Ktrans.txt"))
CBF = np.array(np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/CBF.txt"))
Cell_size = np.array(np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/CellSize.txt"))

mat_fname = "C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/transitmatrix_274625.mat"
mat_contents = sio.loadmat(mat_fname)
transit_matrix = mat_contents["transitmatrix_274625"]