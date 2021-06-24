import numpy as np
from multiprocessing import Pool
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Tissue:
    def __init__(self, tissue_arr3D, cell_size, numthread=-1):
        self.tissue_arr3D = np.transpose(tissue_arr3D,(1,2,0))
        self.Delta_chi = np.array([])
        self.Dfree = 2  # D of free water at room temperature
        self.pad = np.zeros(3)

        self.BC = "periodic"
        self.scheme = "exp"

        self.D = np.ones(3)  # Intrinsic diffusion coefficient 0.7 ~ 1.5

        self.T2 = np.array([200, 200, 200])
        self.T2star = np.array([50, 50, 50])

        self.Pm = np.array(np.ones((3, 3)) * np.spacing(1))

        self.c = np.ones(3)  # water concentration
        self.cfree = 1

        self.n = self.get_tissue_shape()[0]
        self.FOV = np.ones(3) * (self.n * cell_size / 4.5)  # 4.5 is the ardii of the structure
        self.dr = self.FOV / self.n

        self.DR2_micro = np.array([])
        self.DR2_meso = np.array([])
        self.signal = np.array([])
        self.relaxation_matrix = np.array([])
        self.transit_matrix = np.array([])

        self.Ndim = 3

        self.X = np.array([])
        self.Y = np.array([])
        self.Z = np.array([])


    def set_relaxation_matrix(self, relaxation_matrix):
        self.relaxation_matrix = relaxation_matrix

    def get_relaxation_matrix(self):
        return self.relaxation_matrix

    def set_transit_matrix(self, transit_matrix):
        self.transit_matrix = transit_matrix

    def get_transit_matrix(self):
        return self.transit_matrix


    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def set_Z(self, Z):
        self.Z = Z

    def set_DR2_micro(self, DR2_micro):
        self.DR2_micro = DR2_micro

    def set_DR2_meso(self, DR2_meso):
        self.DR2_meso = DR2_meso

    def set_signal(self, signal):
        self.signal = signal

    def get_DR2_micro(self):
        return self.DR2_micro

    def get_DR2_meso(self):
        return self.DR2_meso

    def get_signal(self):
        return self.signal

    def get_tissue(self):
        return self.tissue_arr3D

    def get_cell(self):
        cell_arr3D = np.where(self.tissue_arr3D == 2, 0, self.tissue_arr3D)
        return cell_arr3D

    def get_vascular(self):
        vascular_arr3D = np.where(self.tissue_arr3D == 1, 0, self.tissue_arr3D)
        return vascular_arr3D

    def get_tissue_shape(self):
        return self.tissue_arr3D.shape

    def get_tissue_size(self):
        return self.tissue_arr3D.size

    def get_ratio(self, n):
        return np.count_nonzero(self.tissue_arr3D == n) / self.tissue_arr3D.size

    def vascular_plot3D(self, filename):
        vascular = self.get_vascular()
        z, x, y = vascular.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='blue')
        plt.savefig("out/" + filename)

    def set_Delta_chi(self, concentration_matrix):
        self.Delta_chi = concentration_matrix

    def get_Delta_chi(self):
        return self.Delta_chi

    def set_Dfree(self, D_free_water):
        self.Dfree = D_free_water

# slice1: [x, y, 0] slice2: [x, y, 1]......
# tissue32 = tissue_arr3D[:,:,31]
# print(tissue_info.get_ratio(tissue32, 1))
