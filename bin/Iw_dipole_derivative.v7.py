#!/home/iaw/soft/conda/2024.06.1/envs/pytorch3.9/bin/python
############################################################
## Author: Iaw
## Email: iawhaha@163.com
## Blog: https://iawhome.zicp.fun/
## 这个版本可以追踪水
## 在v5基础上进行修改，主要想增加hann窗，这里增加在求解总偶
## 极自行关函数之后，也就是在偶极自相关函数进行fft求解ir前
############################################################

import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import hydrogenbonds
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from rich import print as rprint
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis import msd
import json
import time
from MDAnalysis.lib.distances import distance_array
from scipy.stats import gaussian_kde
from rich.progress import track
from copy import deepcopy
from scipy import signal
from memory_profiler import profile
import gc
from pympler import asizeof
import tracemalloc
import argparse


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# 所得光谱的分辨率dF是由时间长度决定的，根据采样定理，dF>2/T，T是总时间。 由此我们知道，如果要获得更高分辨率的光谱，需要增长模拟时间。 若需要的光谱分辨率为1 cm^-1左右，那么总时间长度必须大于66 ps，再到相关函数计算时大多只能得到总长度一半的有效数据，那么总时间长度还要加倍，需要140 ps。 这样的模拟时间对经典分子动力学来说不算长，但对从头算或第一原理动力学来说不算短。
# 获得吸收的最高频率是由时间间隔决定的，maxF<1/(2dt)，dt为时间间隔。 一般认为红外光谱的范围为400-4000 cm^-1，因此只要时间间隔小于4 fs即可满足要求。
# v*q 的自相关函数,详细请引用https://doi.org/10.1063/1.3646306
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# 实际上提取的是v*q
# 这里追踪了一直存在的持久水分子
# 这里可以适当简化运算，先循环一次，找到持久水，但是没有做测试，不知道哪个方案跟快速
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def get_V(u, start_frame, end_frame, step, sele_mask, O_charge = False, H_charge = False):
    ow_s = {}
    hw1_s = {}
    hw2_s = {}
    is_first_frame = False
    count = 0
    
    for i_traj in track(u.trajectory[start_frame:end_frame:step]):

        i_v = i_traj.velocities
        seles = u.select_atoms(sele_mask) 

        if is_first_frame == False:
            for sele in seles:
                ow_s[str(sele.resid)] = []
                hw1_s[str(sele.resid)] = []
                hw2_s[str(sele.resid)] = []

                # 开始储存数据
                # 这里是取巧, 水分子储存的时候是按照顺序
                ow_idx = sele.index
                hw1_idx = ow_idx+1
                hw2_idx = ow_idx+2

                ow_s[str(sele.resid)].append(deepcopy(i_v[ow_idx]))
                hw1_s[str(sele.resid)].append(deepcopy(i_v[hw1_idx]))
                hw2_s[str(sele.resid)].append(deepcopy(i_v[hw2_idx]))
                is_first_frame = True
        else:
            resid_seles = {str(sele.resid): [sele.index, sele.index+1, sele.index+2] for sele in seles}

            # 规避 dictionary changed size during iteration
            ow_s_keys = list(ow_s.keys())
            for k in ow_s_keys:
                if k not in resid_seles.keys():
                    # 这里代表的是扩散走的情况 
                    del ow_s[k]
                    del hw1_s[k]
                    del hw2_s[k]
                else:
                    ow_s[k].append(deepcopy(i_v[resid_seles[k][0]]))
                    hw1_s[k].append(deepcopy(i_v[resid_seles[k][1]]))
                    hw2_s[k].append(deepcopy(i_v[resid_seles[k][2]]))
        count += 1

    # 判断数据是否出现致命错误
    _sets = [set(hw1_s.keys()), set(hw2_s.keys()), set(ow_s.keys())]
    all_equal = all(s == _sets[0] for s in _sets) 
    assert all_equal == True, "Fatal Error"
    
    ## 开始合并数据集进行自相关计算
    # atoms, frames, 3 -> frames, atoms, 3
    ow_arr = np.concatenate([[ow_s[k]] for k in ow_s.keys()]).transpose(1, 0, 2)
    hw1_arr = np.concatenate([[hw1_s[k]] for k in hw1_s.keys()]).transpose(1, 0, 2)
    hw2_arr = np.concatenate([[hw2_s[k]] for k in hw2_s.keys()]).transpose(1, 0, 2)
    if O_charge != None:
        ow_arr *= O_charge
        hw1_arr *= H_charge
        hw2_arr *= H_charge

    # 拼接获得最终的矩阵
    final_array = np.concatenate([ow_arr, hw1_arr, hw2_arr], axis=1)

    return final_array

# 主要修订这个函数
# 相较于v4版本，这个地方传入的是所有水分子的加和
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

    u = mda.Universe(fp_top,fp_traj, dt = dt)
    rprint("All number of frames is {}, and the timestep is {:.4f} ps, and the simulation time is {:.4f} ns".format(len(u.trajectory), u.trajectory.dt, len(u.trajectory) * u.trajectory.dt / 1000))
    
    final_array0 = get_V(u, start_frame, end_frame, step, sele_mask, O_charge, H_charge)
    
    # 求解总偶极
    final_array = final_array0.sum(axis = 1)
    
    # 求解总偶极的自相关函数
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





