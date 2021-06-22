import matplotlib.pyplot as plt
import readf as readf
import numpy as np
import DRO as dro
import os


def main():
    """
    "/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/tissue"

    :return:
    """
    file_tissue_dict = {}
    tissue_signal_dict = {}

    file_path = "/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/tissues"
    os.chdir("/Users/yijieguo/PycharmProjects/BNI_Summer2021/DROTestGround/tissues/")
    
    try:
        if file_path.endswith(".txt"):
            readf.read_to_dict(file_tissue_dict, file_path)
        else:
            for entry in os.scandir(file_path):
                if (entry.path.endswith(".txt")) and entry.is_file():
                    readf.read_to_dict(file_tissue_dict, entry.path)
        print("Load Files Successfully!")

        user_interface_filelist(file_tissue_dict, tissue_signal_dict)

    except OSError:
        print("Directory not found.")

    print("Load Files Successfully!")

    # tissue_arr3D = file_tissue_dict["tissueVpVcMets040857_65_65_65.txt"].get_tissue()
    # tissue1 = tissue_arr3D[:,:,20]
    # plt.imshow(tissue1, cmap=plt.cm.gray)
    # plt.savefig("out/" + "2Ddemo.png")


    #user_interface(file_tissue_dict, tissue_signal_dict)

# for tissue in tissue_list:
#   print(tissue.get_tissue())
#  break
# dro.main(tissue)


def user_interface(file_tissue_dict, tissue_signal_dict):
    while True:
        try:
            directory = input("Enter the directory:\n")
            if directory == "-1":
                break
            elif directory.endswith(".txt"):
                readf.read_to_dict(file_tissue_dict, directory)
            else:
                for entry in os.scandir(directory):
                    if (entry.path.endswith(".txt")) and entry.is_file():
                        readf.read_to_dict(file_tissue_dict, entry.path)
            print("Load Files Successfully!")

            user_interface_filelist(file_tissue_dict, tissue_signal_dict)

        except OSError:
            print("Directory not found.")


def user_interface_filelist(file_tissue_dict, tissue_signal_dict):
    while True:
        user_input = input("")

        if user_input == "-1":
            file_tissue_dict.clear()
            tissue_signal_dict.clear()
            break

        elif user_input == "help":
            print(" file: See the list of loaded file. \n"
                  "Enter a file name to start working with one tissue.\n"
                  "-1: Exit")

        elif user_input == "file":
            [print(filename) for filename in file_tissue_dict.keys()]

        elif user_input in file_tissue_dict.keys():
            file = user_input
            tissue = file_tissue_dict[file]
            print("Tissue structure successfully loaded from", file)
            user_interface_tissue(tissue)
        else:
            print("Invalid input! Enter \" help \" to see the list of commands.")


def user_interface_tissue(tissue):
    while True:
        user_input2 = input("")

        if user_input2 == "help":
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
            tissue.vascular_plot3D("vascular3D.png")

        else:
            print("Invalid input! Enter \" help \" to see the list of commands.")


if __name__ == '__main__':
    main()