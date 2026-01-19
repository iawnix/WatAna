from dataclasses import dataclass, field
from .traj_xyz import read_coor_from_xyz, cal_v_from_coor
import numpy as np
from numpy.typing import NDArray
import MDAnalysis as mda
from multiprocessing import Pool
from functools import partial
from typing import List

@dataclass
class pbe_config:
    dt: float
    xyz_f: str
    n_read: int
    qv_save: str

@dataclass
class pbe_mulliken_config:
    dt: float
    vel_f: str
    mulliken_f: str
    n_read: int
    qv_save: str

@dataclass
class pbe_mulliken_r_config:
    dt: float
    xyz_f: str
    mulliken_f: str
    n_read: int
    qr_save: str


@dataclass
class spce_config:
    fp_ncdf: str
    fp_top: str
    start_frame: int
    end_frame: int
    step: int
    sele_mask: str
    atom_charge: List
    n_worker: int
    dt: float
    qv_save: str

@dataclass
class tip3p_config(spce_config):
    pass

@dataclass
class opc3pol_config(spce_config):
    drude: bool


def qv_pbe_mulliken(qv_config) -> NDArray:
    vel_fp = qv_config.vel_f
    mulliken_fp = qv_config.mulliken_f
    n_reads = qv_config.n_read
    dt = qv_config.dt
    qv_save = qv_config.qv_save

    # 格式是一样的,所以这个地方, 可以直接用读取坐标的函数
    # n_frames, n_atoms, 3
    v_s = read_coor_from_xyz(vel_fp, n_reads)
    # n_frames, n_atoms
    q_s = np.load(mulliken_fp)
    qv_s = q_s[:, :, np.newaxis] * v_s
    
    # 所有原子加和
    qv_s_total = qv_s.sum(axis = 1)
    np.save(qv_save, qv_s_total, allow_pickle=True)
    return qv_s_total


def qr_pbe_mulliken(qr_config) -> NDArray:
    xyz_fp = qr_config.xyz_f
    mulliken_fp = qr_config.mulliken_f
    n_reads = qr_config.n_read
    dt = qr_config.dt
    qr_save = qr_config.qr_save

    # 格式是一样的,所以这个地方, 可以直接用读取坐标的函数
    # n_frames, n_atoms, 3
    r_s = read_coor_from_xyz(xyz_fp, n_reads)
    # n_frames, n_atoms
    q_s = np.load(mulliken_fp)
    qr_s = q_s[:, :, np.newaxis] * r_s
    
    # 所有原子加和
    qr_s_total = qr_s.sum(axis = 1)

    np.save(qr_save, qr_s_total, allow_pickle=True)
    return qr_s_total

def qr_pbe_mulliken2(qr_config) -> NDArray:
    xyz_fp = qr_config.xyz_f
    mulliken_fp = qr_config.mulliken_f
    n_reads = qr_config.n_read
    dt = qr_config.dt
    qr_save = qr_config.qr_save

    # 格式是一样的,所以这个地方, 可以直接用读取坐标的函数
    # n_frames, n_atoms, 3
    r_s = read_coor_from_xyz(xyz_fp, n_reads)
    # n_frames, n_atoms
    q_s = np.load(mulliken_fp)

    # 求两次导数
    r_s_ = np.gradient(r_s, dt, axis=0, edge_order=2) 
    q_s_ = np.gradient(q_s, dt, axis=0, edge_order=2) 

    u_s_ = q_s_[:, :, np.newaxis] * r_s + q_s[:, :, np.newaxis] * r_s_
    
    # 所有原子加和
    u_s_total_ = u_s_.sum(axis = 1)

    np.save(qr_save, u_s_total_, allow_pickle=True)
    return u_s_total_


def qv_pbe(qv_config) -> NDArray:
    xyz_fp = qv_config.xyz_f
    n_reads = qv_config.n_read
    dt = qv_config.dt
    qv_save = qv_config.qv_save
    # 其实读取的会多1, 这样输出的速度就会是是原来的n_reads
    n_reads += 1

    coor_s = read_coor_from_xyz(xyz_fp, n_reads)
    v_s = cal_v_from_coor(coor_s, dt)

    # 这里生成一个q的张量, 形状等同于n_frames, n_atoms, 3
    q_s = np.zeros(shape = v_s.shape)

    for i in range(0, v_s.shape[1], 3):
        q_s[:, i, :] = -2     # O 原子
        q_s[:, i+1, :] = 1  # H 原子
        q_s[:, i+2, :] = 1  # H 原子

    qv_s = v_s * q_s

    # 所有原子加和
    qv_s_total = qv_s.sum(axis = 1)
    np.save(qv_save, qv_s_total, allow_pickle=True)
    return qv_s_total


def __get_qv_spce__(i_frame, u, sele_mask, O_charge = False, H_charge = False) -> NDArray:
    total_u_ = np.zeros(shape=(3,))

    i_traj = u.trajectory[i_frame]
    i_v = i_traj.velocities
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


def qv_spce(qv_config) -> NDArray:
    fp_ncdf = qv_config.fp_ncdf
    fp_top = qv_config.fp_top
    start_frame = qv_config.start_frame
    end_frame = qv_config.end_frame
    step = qv_config.step
    sele_mask = qv_config.sele_mask
    O_charge, H_charge = qv_config.atom_charge
    n_worker = qv_config.n_worker
    dt = qv_config.dt
    qv_save = qv_config.qv_save

    u = mda.Universe(fp_top,fp_ncdf, dt = dt)
    print("All number of frames is {}, and the timestep is {:.4f} ps, and the simulation time is {:.4f} ns".format(len(u.trajectory)
                                , u.trajectory.dt
                                , len(u.trajectory) * u.trajectory.dt / 1000))

    single_frame = partial(__get_qv_spce__
                , u = u
                , sele_mask = sele_mask
                , O_charge = O_charge, H_charge = H_charge)
    
    frames = list(range(start_frame,end_frame,step))

    # 这种任务, 实际上并行的效率并不是很高, 一定要注意内存
    with Pool(n_worker) as worker_pool:
        result = worker_pool.map(single_frame, frames)

    qv_s_total = np.stack(result, axis = 0)
    print(qv_s_total.shape)
    np.save(qv_save, qv_s_total, allow_pickle=True)
    return qv_s_total

def qv_tip3p(qv_config) -> NDArray:
    # 两者都是三点水模型, 可以通用
    return qv_spce(qv_config)


def __get_qv_opc3pol_drude__(i_frame, u, sele_mask, O_charge = False, H_charge = False, Y_charge = False) -> NDArray:
    total_u_ = np.zeros(shape=(3,))

    i_traj = u.trajectory[i_frame]
    i_v = i_traj.velocities
    seles = u.select_atoms(sele_mask) 

    # 这里是取巧, 水分子储存的时候是按照顺序
    ow_idx = seles.indices
    hw1_idx = seles.indices+1
    hw2_idx = seles.indices+2
    y1_idx = seles.indices+3
    #n_atm, 3 -> 3, 1
    qv_ow = i_v[ow_idx].sum(axis = 0)*O_charge
    qv_hw1 = i_v[hw1_idx].sum(axis = 0)*H_charge
    qv_hw2 = i_v[hw2_idx].sum(axis = 0)*H_charge
    qv_y1 = i_v[y1_idx].sum(axis = 0)*Y_charge
    total_u_ += (qv_ow + qv_hw1 + qv_hw2 + qv_y1)
    return total_u_

def __get_qv_opc3pol__(i_frame, u, sele_mask, O_charge = False, H_charge = False, Y_charge = False) -> NDArray:
    total_u_ = np.zeros(shape=(3,))

    i_traj = u.trajectory[i_frame]
    i_v = i_traj.velocities
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

def qv_opc3pol(qv_config) -> NDArray:
    fp_ncdf = qv_config.fp_ncdf
    fp_top = qv_config.fp_top
    start_frame = qv_config.start_frame
    end_frame = qv_config.end_frame
    step = qv_config.step
    sele_mask = qv_config.sele_mask
    O_charge, H_charge, Y1_charge = qv_config.atom_charge
    n_worker = qv_config.n_worker
    dt = qv_config.dt
    qv_save = qv_config.qv_save
    drude = qv_config.drude 

    u = mda.Universe(fp_top,fp_ncdf, dt = dt)
    print("All number of frames is {}, and the timestep is {:.4f} ps, and the simulation time is {:.4f} ns".format(len(u.trajectory)
                                , u.trajectory.dt
                                , len(u.trajectory) * u.trajectory.dt / 1000))

    if drude:
        single_frame = partial(__get_qv_opc3pol_drude__
                    , u = u
                    , sele_mask = sele_mask
                    , O_charge = O_charge, H_charge = H_charge, Y_charge = Y1_charge)
    else:
        single_frame = partial(__get_qv_opc3pol__
                    , u = u
                    , sele_mask = sele_mask
                    , O_charge = O_charge, H_charge = H_charge, Y_charge = Y1_charge)
    frames = list(range(start_frame,end_frame,step))

    # 这种任务, 实际上并行的效率并不是很高, 一定要注意内存
    with Pool(n_worker) as worker_pool:
        result = worker_pool.map(single_frame, frames)

    qv_s_total = np.stack(result, axis = 0)
    print(qv_s_total.shape)
    np.save(qv_save, qv_s_total, allow_pickle=True)
    return qv_s_total
