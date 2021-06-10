import DRO as dro
import numpy as np

global tissue


def main():
    dir = "/Users/user/PycharmProjects/yijieDRO/tissue/tissueVpVcMets040857_65_65_65.txt"
    tissue = dro.DRO(dir)
    print(tissue.get_tissue())
    print(tissue.get_tissue_size())


if __name__ == '__main__':
    main()
