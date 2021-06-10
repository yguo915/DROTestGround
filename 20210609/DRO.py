import readf as readf
import numpy as np


class DRO:
    def __init__(self, dir, numthread=-1):
        self.tissue_arr3D = readf.read_file(dir)

    def get_tissue(self):
        return self.tissue_arr3D

    def get_tissue_size(self):
        return self.tissue_arr3D.shape
