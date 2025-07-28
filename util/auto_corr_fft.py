import numpy as np
from scipy import signal
from typing import List
from numpy.typing import NDArray

def auto_corr_fft(arr_3d: NDArray, fraction_autocorrelation_function_to_fft: int = 0.1) -> NDArray:

    # arr_3d: n_frames,  3
    arr_x = arr_3d[:, 0]
    arr_y = arr_3d[:, 1]
    arr_z = arr_3d[:, 2]

    # 此处代码借鉴https://github.com/EfremBraun/calc-ir-spectra-from-lammps/blob/master/calc-ir-spectra.py
    time_len  = arr_3d.shape[0]

    if  time_len % 2 == 0:
            arr_x_shifted = np.zeros(time_len*2)
            arr_y_shifted = np.zeros(time_len*2)
            arr_z_shifted = np.zeros(time_len*2)
    else:
            arr_x_shifted = np.zeros(time_len*2-1)
            arr_y_shifted = np.zeros(time_len*2-1)
            arr_z_shifted = np.zeros(time_len*2-1)

    arr_x_shifted[time_len//2:time_len//2+time_len] = arr_x
    arr_y_shifted[time_len//2:time_len//2+time_len] = arr_y
    arr_z_shifted[time_len//2:time_len//2+time_len] = arr_z

    # Convolute the shifted array with the flipped array, which is equivalent to performing a correlation
    autocorr_x_full = (signal.fftconvolve(arr_x_shifted, arr_x[::-1], mode='same')[(-time_len):]
                       / np.arange(time_len, 0, -1))
    autocorr_y_full = (signal.fftconvolve(arr_y_shifted, arr_y[::-1], mode='same')[(-time_len):]
                       / np.arange(time_len, 0, -1))
    autocorr_z_full = (signal.fftconvolve(arr_z_shifted, arr_z[::-1], mode='same')[(-time_len):]
                       / np.arange(time_len, 0, -1))

    # 偶极矩导数是一个矢量，表示分子或系统的电荷分布特征。在计算偶极矩导数自相关函数时，我们关心的是偶极矩随时间的变化情况，而不仅仅是某个特定方向的变化。偶极矩的总自相关函数应该反映偶极矩在所有方向上的变化特征。
    autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full

    # Truncate the autocorrelation array
    autocorr = autocorr_full[:int(time_len * fraction_autocorrelation_function_to_fft)]

    # 归一化
    autocorr /= autocorr[0]

    return autocorr

if __name__ == "__main__":
    # Test
    print("This is some test")
