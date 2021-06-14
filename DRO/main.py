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
            print("Load Files Successfully!")

            while True:
                user_input = input("")

                if user_input == "-1":
                    file_list = []
                    tissue_list = []
                    signal_list = []
                    break

                elif user_input == "help":
                    print(" file: See the list of loaded file. \n"
                          "Enter a file name to start working with one tissue.\n"
                          "-1: Exit")

                elif user_input == "file":
                    [print(filename) for filename in file_list]

                elif user_input in file_list:
                    file = user_input
                    tissue = tissue_list[file_list.index(file)]
                    print("Tissue structure successfully loaded from", file)
                    while True:
                        user_input2 = input("")

                        if  user_input2 == "help":
                            print(" tissue: To see current tissue. \n"
                                  "-1: Exit")

                        elif user_input2 == "-1":
                            print("Back to list of files.")
                            break

                        elif user_input2 == "tissue":
                            print(tissue.get_tissue_shape())
                            print(tissue.get_tissue())

                        elif user_input2 == "tissue ratio":
                            print("0:", tissue.get_ratio(0))
                            print("1:", tissue.get_ratio(1))
                            print("2:", tissue.get_ratio(2))

                        elif user_input2 == "plot":
                            tissue.vascular_plot3D("vascular.png")

                        else:
                            print("Invalid input! Enter \" help \" to see the list of commands.")

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