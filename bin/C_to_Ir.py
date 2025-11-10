import numpy as np
from scipy.signal import savgol_filter, find_peaks
from numpy.fft import fft, fftfreq
from numpy.typing import NDArray



def cal_ir(qv_fp: str):

    qv_autocoor_sim = np.loadtxt(qv_fp, delimiter=",")
        
    def _hann(signal: NDArray):


    def C_dipole(x, y):

        dt_ps = x[1] - x[0]     # ps
        N = len(x)              #
        spectrum = fft(y).real[:N//2]
        freq = fftfreq(N, d = dt_ps)[:N//2]*33  # Thz -> cm-1
        return freq, spectrum

    freq_sim, spectrum_sim = C_dipole(qv_autocoor_sim[:,0], qv_autocoor_sim[:,1])

    with open("Iw_smi_SP_{}_40ps.csv".format(flag), "w+") as F:
        for i in range(spectrum_sim.shape[0]):
            F.writelines("{:.6f},{:.6f}\n".format(freq_sim[i], spectrum_sim[i]))


