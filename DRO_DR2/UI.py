import tkinter as tk
from tkinter.filedialog import askopenfilename

import numpy as np
import cupy as cp
import os
import scipy.io as sio

import Tissue as t
import Pulse as p
import readf as readf
import DR2 as dr2
import concentrationMatrix as cm

class UI(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.master.title("DSC-MRI DRO")
        self.create_widgets()


    def create_widgets(self):
        self.title = tk.Label(self.master, text = "MRI Digital Reference Object(DRO)", fg = "black", width = 80, height = 5, font = ("Arial", 25))

        self.upload_tissue_button = tk.Button(self.master, text = "upload tissue",font = ("Arial", 18), command = self.upload_file)

        # self.bar = tk.Label(self.master, text="####################################################################################)", fg="black", width=80, height=5,
        #                       font=("Arial", 12))

        self.title.pack()
        self.upload_tissue_button.pack()
        # self.bar.pack()

    def upload_file(self):
        dir = os.getcwd()
        os.chdir(dir)
        arterial_input_function_full = np.array(np.loadtxt(dir+
            "/input/aiffull_modified.txt"))
        Ktrans = np.array(np.loadtxt(dir+"/input/Ktrans.txt"))
        CBF = np.array(np.loadtxt(dir+"/input/CBF.txt"))
        Cell_size = np.array(np.loadtxt(dir+"/input/CellSize.txt"))
        mat_contents = sio.loadmat(dir+"/input/transitmatrix_274625.mat")
        transit_matrix = mat_contents["transitmatrix_274625"]


        file_path =  askopenfilename()
        PP = 0
        NR = 0
        tissue = t.Tissue(readf.read_txt_to_arr3D(file_path), Cell_size[NR])
        pulse = p.Pulse()
        #cm.concentration_matrix(tissue, CBF, Ktrans, arterial_input_function_full)

        print(tissue.get_tissue_shape())
        #ot.main(tissue, arterial_input_function_full, Ktrans, CBF, PP, NR)
        # print(DR2.tissue.get_DR2_total())
        DR2 = dr2.DRO_DR2(tissue, pulse, arterial_input_function_full, Ktrans, CBF, transit_matrix, NR, PP)
        DR2.get_DR2_total()

if __name__ == '__main__':
    global tissue
    root = tk.Tk()
    app = UI(master = root)
    app.mainloop()


