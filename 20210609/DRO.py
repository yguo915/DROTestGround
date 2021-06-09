import readf as rf
import numpy as np


class DRO:
    def __init__(self, dir, numthread=-1):
        self.arr = rf.readf(dir)

    def getTissue(self):
        return self.arr

    def getTissueSize(self):
        return self.arr.shape
