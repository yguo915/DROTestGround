import Tissue as t
import readf as readf
import numpy as np
import DRO as dro
import os



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
                readf.add_to_list(tissue_list, file_list, directory)
            else:
                for entry in os.scandir(directory):
                    if (entry.path.endswith(".txt")) and entry.is_file():
                        readf.add_to_list(tissue_list, file_list, entry.path)
            break
        except OSError:
            print("Directory not found.")



   # for tissue in tissue_list:
     #   print(tissue.get_tissue())
      #  break
        # dro.main(tissue)


if __name__ == '__main__':
    main()
