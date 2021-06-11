import numpy as np
from multiprocessing import Pool
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Tissue:
    def __init__(self, tissue_arr3D, numthread=-1):
        self.tissue_arr3D = tissue_arr3D

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

    def vascular_plot3D(self,filename):
        vascular = self.get_vascular()
        z,x,y = vascular.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='blue')
        plt.savefig(filename)

# slice1: [x, y, 0] slice2: [x, y, 1]......
# tissue32 = tissue_arr3D[:,:,31]
# print(tissue_info.get_ratio(tissue32, 1))
