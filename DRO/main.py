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
            if directory == "-1":
                break
            elif directory.endswith(".txt"):
                tissue_list.append(t.Tissue(readf.read_file(directory)))
                readf.add_to_list(tissue_list, file_list, directory)
            else:
                for entry in os.scandir(directory):
                    if (entry.path.endswith(".txt")) and entry.is_file():
                        readf.add_to_list(tissue_list, file_list, entry.path)
            print("Load Files Successfully! Enter \" help \" to see the list of commands.")

            while True:
                user_input = input("")
                if user_input == "-1":
                    break

                elif user_input == "help":
                    print(" file: See the list of loaded file. \n"
                          "-1: Exit")

                elif user_input == "file":
                    [print(filename) for filename in file_list]

                else:
                    print("Invalid input! Enter \" help \" to see the list of commands.")

        except OSError:
            print("Directory not found.")


# for tissue in tissue_list:
#   print(tissue.get_tissue())
#  break
# dro.main(tissue)


if __name__ == '__main__':
    main()
