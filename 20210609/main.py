import DRO as dro
import readf as readf
import tissue_info as tissue_info
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

global tissue


def main():
    dir = "/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/tissue/tissueVpVcMets040857_65_65_65.txt"

    tissue_arr3D = readf.read_file(dir)
    tissue = dro.DRO(tissue_arr3D)
    
    vascular = tissue.get_vascular()

    z,x,y = vascular.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='blue')
    plt.savefig("vascular_demo.png")




if __name__ == '__main__':
    main()
