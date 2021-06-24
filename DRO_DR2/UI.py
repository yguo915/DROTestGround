import tkinter as tk
from tkinter.filedialog import askopenfilename

import numpy as np
import cupy as cp
import os

import Tissue as t
import readf as readf
import oneTissue as ot

class UI(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.master.title("DSC-MRI DRO")
        self.create_widgets()


    def create_widgets(self):
        self.title = tk.Label(self.master, text = "MRI Digital Reference Object(DRO)", fg = "black", width = 80, height = 5, font = ("Arial", 25))

        self.upload_tissue_button = tk.Button(self.master, text = "upload tissue",font = ("Arial", 18), command = self.upload_file)

        self.bar = tk.Label(self.master, text="####################################################################################)", fg="black", width=80, height=5,
                              font=("Arial", 12))

        self.title.pack()
        self.upload_tissue_button.pack()
        self.bar.pack()

    def upload_file(self):
        arterial_input_function_full = np.array(np.loadtxt(
            "C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/aiffull_modified.txt"))
        Ktrans = np.array(np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/Ktrans.txt"))
        CBF = np.array(np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/CBF.txt"))
        Cell_size = np.array(np.loadtxt("C:/Users/user/PycharmProjects/yijieDRO/DROTestGround/DROTestGround/input/CellSize.txt"))


        file_path =  askopenfilename()
        PP = 0
        NR = 0
        global tissue
        tissue = t.Tissue(readf.read_txt_to_arr3D(file_path), Cell_size[NR])
        print(tissue.get_tissue_shape())
        ot.main(tissue, arterial_input_function_full, Ktrans, CBF, PP, NR)

if __name__ == '__main__':
    root = tk.Tk()
    app = UI(master = root)
    app.mainloop()


