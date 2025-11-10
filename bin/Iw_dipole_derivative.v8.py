#!/home/iaw/soft/conda/2024.06.1/envs/pytorch3.9/bin/python
############################################################
## Author: Iaw
## Email: iawhaha@163.com
## Blog: https://iawhome.zicp.fun/
## 这个版本是不追踪水的
## 继承v6，主要更新是增加hann
############################################################

import MDAnalysis as mda
import sys
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rich import print as rprint
import json
import time
from rich.progress import track
from copy import deepcopy
from scipy import signal
import argparse
from functools import partial
from multiprocessing import Pool
from rich.status import Status


# 所得光谱的分辨率dF是由时间长度决定的，根据采样定理，dF>2/T，T是总时间。 由此我们知道，如果要获得更高分辨率的光谱，需要增长模拟时间。 若需要的光谱分辨率为1 cm^-1左右，那么总时间长度必须大于66 ps，再到相关函数计算时大多只能得到总长度一半的有效数据，那么总时间长度还要加倍，需要140 ps。 这样的模拟时间对经典分子动力学来说不算长，但对从头算或第一原理动力学来说不算短。
# 获得吸收的最高频率是由时间间隔决定的，maxF<1/(2dt)，dt为时间间隔。 一般认为红外光谱的范围为400-4000 cm^-1，因此只要时间间隔小于4 fs即可满足要求。
# v*q 的自相关函数,详细请引用https://doi.org/10.1063/1.3646306

# 区别于v5 这里相当于是某一个区域的总偶极
# 区别于v6, 这里是并行版本
def get_QV(i_frame, u,sele_mask, O_charge = False, H_charge = False) -> NDArray:

    total_u_ = np.zeros(shape=(3,))

    i_traj = u.trajectory[i_frame]
    i_v = i_traj.velocities

    # 这里会重新update sele，相当于固定了一个区域，尽管不太严谨，因为水化层会一直波动
    seles = u.select_atoms(sele_mask)

    # 这里是取巧, 水分子储存的时候是按照顺序
    ow_idx = seles.indices
    hw1_idx = seles.indices+1
    hw2_idx = seles.indices+2

    #n_atm, 3 -> 3, 1
    qv_ow = i_v[ow_idx].sum(axis = 0)*O_charge
    qv_hw1 = i_v[hw1_idx].sum(axis = 0)*H_charge
    qv_hw2 = i_v[hw2_idx].sum(axis = 0)*H_charge
    total_u_ += (qv_ow + qv_hw1 + qv_hw2)

    return total_u_


# 主要修订这个函数
def QV_autocorr_fft(qv_s_total, fraction_autocorrelation_function_to_fft = 0.1):
    out_fft = []

    # vq_s_total: n_frames,  3
    dipole_x = qv_s_total[:, 0]
    dipole_y = qv_s_total[:, 1]
    dipole_z = qv_s_total[:, 2]
    
    # 此处代码借鉴https://github.com/EfremBraun/calc-ir-spectra-from-lammps/blob/master/calc-ir-spectra.py
    # 采用fft加速计算
    time_len  = qv_s_total.shape[0]
    
    if  time_len % 2 == 0:
            dipole_x_shifted = np.zeros(time_len*2)
            dipole_y_shifted = np.zeros(time_len*2)
            dipole_z_shifted = np.zeros(time_len*2)
    else:
            dipole_x_shifted = np.zeros(time_len*2-1)
            dipole_y_shifted = np.zeros(time_len*2-1)
            dipole_z_shifted = np.zeros(time_len*2-1)
    
    dipole_x_shifted[time_len//2:time_len//2+time_len] = dipole_x
    dipole_y_shifted[time_len//2:time_len//2+time_len] = dipole_y
    dipole_z_shifted[time_len//2:time_len//2+time_len] = dipole_z
    
    # Convolute the shifted array with the flipped array, which is equivalent to performing a correlation
    autocorr_x_full = (signal.fftconvolve(dipole_x_shifted, dipole_x[::-1], mode='same')[(-time_len):]
                       / np.arange(time_len, 0, -1))
    autocorr_y_full = (signal.fftconvolve(dipole_y_shifted, dipole_y[::-1], mode='same')[(-time_len):]
                       / np.arange(time_len, 0, -1))
    autocorr_z_full = (signal.fftconvolve(dipole_z_shifted, dipole_z[::-1], mode='same')[(-time_len):]
                       / np.arange(time_len, 0, -1))

    # 偶极矩导数是一个矢量，表示分子或系统的电荷分布特征。在计算偶极矩导数自相关函数时，我们关心的是偶极矩随时间的变化情况，而不仅仅是某个特定方向的变化。偶极矩的总自相关函数应该反映偶极矩在所有方向上的变化特征。
    autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full
    # Truncate the autocorrelation array
    autocorr = autocorr_full[:int(time_len * fraction_autocorrelation_function_to_fft)]
    
    # 归一化
    autocorr /= autocorr[0]

    return autocorr



def Parm():
    parser = argparse.ArgumentParser(description=
                                     "The author is very lazy and doesn't want to write anything\n"
                                     "Author: IAWNIX [ECNU]"
                                    )
    parser.add_argument("-top",type=str, nargs=1, help="FilePath, Format: XXX.xx")
    parser.add_argument("-ncdf",type=str, nargs=1, help="FilePath, Format: XXX.xx")
    parser.add_argument("-dt",type=str, nargs=1, help="ps")
    parser.add_argument("-SES",type=str, nargs=1, help="Start:End:Step")
    parser.add_argument("-Oe",type=str, nargs=1, help="")
    parser.add_argument("-He",type=str, nargs=1, help="")
    parser.add_argument("-frac",type=str, nargs=1, help="fraction_autocorrelation_function_to_fft: defaut = 0.1")
    parser.add_argument("-outf",type=str, nargs=1, help="outf path")
    parser.add_argument("-seles",type=str, nargs=1, help="[mdanalysis sele]")
    parser.add_argument("-ncpu",type=str, nargs=1, help="The number of cpu")
    parser.add_argument("-hann",type=bool, nargs=1, help="[hann window]")
    return parser.parse_args()





if __name__ == "__main__":


    HELLO = """
            ██╗ █████╗ ██╗    ██╗
            ██║██╔══██╗██║    ██║
            ██║███████║██║ █╗ ██║
            ██║██╔══██║██║███╗██║
            ██║██║  ██║╚███╔███╔╝
            ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ 
            """
    rprint(HELLO)

    myP = Parm()
    
    fp_top = myP.top[0]
    fp_traj = myP.ncdf[0]
    dt = eval(myP.dt[0])

    start_frame, end_frame, step = [eval(i) for i in myP.SES[0].split(":")]
    O_charge = eval(myP.Oe[0])
    H_charge = eval(myP.He[0])
    fraction_autocorrelation_function_to_fft = eval(myP.frac[0])
    v_autocorr_out = myP.outf[0]
    sele_mask = myP.seles[0]
    hann = myP.hann[0]
    n_work = eval(myP.ncpu[0])

    u = mda.Universe(fp_top,fp_traj, dt = dt)
    rprint("All number of frames is {}, and the timestep is {:.4f} ps, and the simulation time is {:.4f} ns".format(len(u.trajectory), u.trajectory.dt, len(u.trajectory) * u.trajectory.dt / 1000))
    
    # 这里已经是总偶极
    with Status("[cyan]Processing frames...") as status:
        single_frame = partial(get_QV
                    , u = u
                    , sele_mask = sele_mask
                    , O_charge = O_charge, H_charge = H_charge)

        frames = list(range(start_frame,end_frame,step))
        with Pool(n_work) as worker_pool:
            result = worker_pool.map(single_frame, frames)
        
        final_array = np.stack(result, axis = 0)
    
    # 计算自相关
    final_autocorr = QV_autocorr_fft(final_array, fraction_autocorrelation_function_to_fft)

    time_step = step * dt                                        # ps
    final_autocorr_x = np.array(list(range(final_autocorr.shape[0])), dtype=np.float64)
    final_autocorr_x *= time_step                                # ps

    if hann:
        n = np.arange(N:=len(final_autocorr))
        hann_win = (np.cos(np.pi*n / (N-1))) **2
        fina_autocorr_hann = hann_win * final_autocorr
        with open(v_autocorr_out.replace(".csv", "_hann.csv"), "w+") as F:
            for i, i_x in enumerate(final_autocorr_x):
                F.writelines("{:.6f}, {:.6f}\n".format(i_x, final_autocorr[i]))
 
    # save
    with open(v_autocorr_out, "w+") as F:
        for i, i_x in enumerate(final_autocorr_x):
            F.writelines("{:.6f}, {:.6f}\n".format(i_x, final_autocorr[i]))





