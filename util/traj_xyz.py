from typing import List
import numpy as np
from numpy.typing import NDArray
from scipy import signal
import pickle

def read_coor_from_xyz(fp: str, n_frames: int) -> NDArray:

    with open(fp, "r") as F:

        lines = F.readlines()
        n_atm = eval(lines[0].rstrip("\n").replace(" ", ""))


        coor_s = np.zeros(shape=(n_frames,n_atm, 3))

        for i in range(n_frames):

            start_index = 2+(n_atm+2)*i

            end_index = start_index+ n_atm
            if end_index == len(lines):
                end_index = -1

            i_xyz_s: List = lines[start_index:end_index]

            for j, i_xyz in enumerate(i_xyz_s):
                _var = i_xyz.rstrip("\n").split()
                coor_s[i, j, 0] += eval(_var[1])
                coor_s[i, j, 1] += eval(_var[2])
                coor_s[i, j, 2] += eval(_var[3])

    return coor_s

def cal_v_from_coor(coor_s, dt) -> NDArray:

    # coor_s : n_frames, n_atoms, 3
    v_s = np.diff(coor_s, axis=0) / dt
    
    return v_s


