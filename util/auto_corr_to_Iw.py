from numpy.fft import fft, fftfreq
import numpy as np
from numpy.typing import NDArray
from typing import Tuple



def auto_corr_hann(arr_1D: NDArray) -> NDArray:
    # 生成Hann window
    n = np.arange(N:=len(arr_1D))

    # cos^2(pi*n / (N-1))
    
    hann_win = (np.cos(np.pi*n / (N-1))) **2

    return hann_win * arr_1D

def fft_to_Iw(t: NDArray, y: NDArray, t_unit: str = "ps") -> Tuple[NDArray, NDArray]:
    
    assert t_unit == "ps", print("You must ensure the unit of time is ps!")
    dt_ps = t[1] - t[0]
    N = len(t)
    
    spectrum = fft(y).real[:N//2]
    freq = fftfreq(N, d = dt_ps)[:N//2] * 33

    return (freq, spectrum)


if __name__ == "__main__":

    # test
    print("This is test.")
