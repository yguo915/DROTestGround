import Tissue as t
import readf as readf
import numpy as np
import DRO as dro
import os
import ntpath


def main():
    """
    "/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/tissue"

    :return:
    """
    file_list = []
    tissue_list = []
    signal_list = []

    while True:
        try:
            directory = input("Enter the directory:\n")
            if directory.endswith(".txt"):
                tissue_list.append(t.Tissue(readf.read_file(directory)))
                add_to_list(tissue_list, file_list, directory)
            else:
                for entry in os.scandir(directory):
                    if (entry.path.endswith(".txt")) and entry.is_file():
                        add_to_list(tissue_list, file_list, entry.path)
            break
        except OSError:
            print("Directory not found.")

    for tissue in tissue_list:
        print(tissue.get_tissue())
        break
        # dro.main(tissue)


def add_to_list(tissue_list, file_list, path):
    tissue_list.append(t.Tissue(readf.read_file(path)))
    file_list.append(ntpath.basename(path))


if __name__ == '__main__':
    main()
