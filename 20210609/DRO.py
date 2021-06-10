import numpy as np
from multiprocessing import Pool
from itertools import product


class DRO:
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