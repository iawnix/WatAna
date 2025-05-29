############################################################
## Author: Iaw
## Email: iawhaha@163.com
##Blog: https://iawhome.zicp.fun/
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

#
# 所得光谱的分辨率dF是由时间长度决定的，根据采样定理，dF>2/T，T是总时间。 由此我们知道，如果要获得更高分辨率的光谱，需要增长模拟时间。 若需要的光谱分辨率为1 cm^-1左右，那么总时间长度必须大于66 ps，再到相关函数计算时大多只能得到总长度一半的有效数据，那么总时间长度还要加倍，需要140 ps。 这样的模拟时间对经典分子动力学来说不算长，但对从头算或第一原理动力学来说不算短。
# 获得吸收的最高频率是由时间间隔决定的，maxF<1/(2dt)，dt为时间间隔。 一般认为红外光谱的范围为400-4000 cm^-1，因此只要时间间隔小于4 fs即可满足要求。
# v*q 的自相关函数,详细请引用https://doi.org/10.1063/1.3646306


## 这里可以进一步简化运算
## 只需要读取第一帧与最后一帧做一个差就可以
## 实际上提取的是v*q
def get_V(u, start_frame, end_frame, step, sele_mask, O_charge = False, H_charge = False):
    ow_s = {}
    hw1_s = {}
    hw2_s = {}
    is_first_frame = False
    count = 0
    #tracemalloc.start()
    for i_traj in track(u.trajectory[start_frame:end_frame:step]):
    #for i, i_traj in track(enumerate(u.trajectory[10:15:1])):

        #start_time = time.time()
        # 获取所有的速度数据
        # 需要copy一份, 要不数据会被覆盖
        #i_v = deepcopy(i_traj.velocities)
        i_v = i_traj.velocities
        #print(i_v[1275])
        #seles = u.select_atoms("(around 4 ({}) ) and type OW".format("resname IAH"))    
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
            #print(resid_seles)
            # 规避 dictionary changed size during iteration
            ow_s_keys = list(ow_s.keys())
            for k in ow_s_keys:
                if k not in resid_seles.keys():
                    # 这里代表的是扩散走的情况 
                    #_a = ow_s[k]
                    #print(gc.get_referrers(ow_s[k]))
                    #current_memory = tracemalloc.get_traced_memory()
                    #print("iaw ow_s[k], ref:",sys.getrefcount(ow_s[k]), "size: (bytes)", asizeof.asizeof(ow_s[k]),"k in ow_s", k in ow_s.keys(), "curr_mem: {} bytes".format(current_memory[0]))
                    #print("iaw ow_s[k], ref:", sys.getrefcount(ow_s[k]), "size: (bytes)", asizeof.asizeof(ow_s),"k in ow_s", k in ow_s.keys(), end = "\t")
                    del ow_s[k]
                    del hw1_s[k]
                    del hw2_s[k]
                    #print(k in ow_s.keys(), "_a mem: ", asizeof.asizeof(_a))
                    #current_memory = tracemalloc.get_traced_memory()
                    #print("k in ow_s", k in ow_s.keys(), "curr_mem: {} bytes".format(current_memory[0]))
                    #print("k in ow_s", k in ow_s.keys(), "size: (bytes)", asizeof.asizeof(ow_s))
                else:
                    #print(i_v[resid_seles[k][0]], resid_seles[k][0], k)
                    ow_s[k].append(deepcopy(i_v[resid_seles[k][0]]))
                    hw1_s[k].append(deepcopy(i_v[resid_seles[k][1]]))
                    hw2_s[k].append(deepcopy(i_v[resid_seles[k][2]]))
        #print("iaw i_v, ref:",sys.getrefcount(i_v), "size: (bytes)", asizeof.asizeof(i_v), end = "\t")
        #del i_v
        #print("frame{}".format(count) , asizeof.asizeof(ow_s), end = "\n")
        #gc.collect()
        #print("after gc", asizeof.asizeof(ow_s))
        #end_time = time.time()
        #print("frame{}, time: {:.5f} s".format(count, end_time - start_time))
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

def V_autocorr_fft(final_array, fraction_autocorrelation_function_to_fft = 0.1):
    out_fft = []
    for i in range(final_array.shape[1]):
        dipole_x = final_array[:, i, 0]
        dipole_y = final_array[:, i, 1]
        dipole_z = final_array[:, i, 2]
    
        # 此处代码借鉴https://github.com/EfremBraun/calc-ir-spectra-from-lammps/blob/master/calc-ir-spectra.py
        # 采用fft加速计算
        time_len  = final_array.shape[0]
    
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
        autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full
        # Truncate the autocorrelation array
        autocorr = autocorr_full[:int(time_len * fraction_autocorrelation_function_to_fft)]
        out_fft.append(autocorr)
    
    # 平均并归一化
    final_autocorr = np.array(out_fft).mean(axis=0)
    final_autocorr /= final_autocorr[0]

    return final_autocorr



def Parm():
    parser = argparse.ArgumentParser(description=
                                     "The author is very lazy and doesn't want to write anything\n"
                                     "Author: IAW [ECNU]"
                                    )
    parser.add_argument("-top",type=str, nargs=1, help="FilePath, Format: XXX.xx")
    parser.add_argument("-ncdf",type=str, nargs=1, help="FilePath, Format: XXX.xx")
    parser.add_argument("-dt",type=str, nargs=1, help="")
    parser.add_argument("-SES",type=str, nargs=1, help="")
    parser.add_argument("-Oe",type=str, nargs=1, help="")
    parser.add_argument("-He",type=str, nargs=1, help="")
    parser.add_argument("-frac",type=str, nargs=1, help="fraction_autocorrelation_function_to_fft")
    parser.add_argument("-out",type=str, nargs=1, help="outf path")
    parser.add_argument("-seles",type=str, nargs=1, help="[mdanalysis sele]")

    parser.add_argument("-qm",type=str, nargs=1, help="Only output information for the QM area: y or n")
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
    
    fp_top = "../model.prmtop"                 # 
    fp_traj = "../md-b10w-50ps.ncdf"
    dt = 0.0005                                             # ps
    start_frame = 0
    end_frame = 80000
    step = 1
    O_charge = -0.8400
    H_charge = 0.4200
    fraction_autocorrelation_function_to_fft = 0.1
    v_autocorr_out = "./v_autocorr_mc.csv"
    sele_mask = "(around 4 ({}) ) and type OW".format("resid 1-898")
    u = mda.Universe(fp_top,fp_traj, dt = dt)
    rprint("All number of frames is {}, and the timestep is {:.4f}ps, and the simulation time is {:.4f}ns".format(len(u.trajectory), u.trajectory.dt, len(u.trajectory) * u.trajectory.dt / 1000))
    
    final_array = get_V(u, start_frame, end_frame, step, sele_mask, O_charge, H_charge)
    final_autocorr = V_autocorr_fft(final_array, fraction_autocorrelation_function_to_fft)

    time_step = step * dt                              # ps
    final_autocorr_x = np.array(list(range(final_autocorr.shape[0])), dtype=np.float64)
    final_autocorr_x *= time_step              #ps

    # save
    with open(v_autocorr_out, "w+") as F:
        for i, i_x in enumerate(final_autocorr_x):
            F.writelines("{:.6f}, {:.6f}\n".format(i_x, final_autocorr[i]))





