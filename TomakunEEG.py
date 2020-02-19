import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QListWidget, QApplication, QListWidgetItem, QFileDialog, \
    QGridLayout, \
    QTableWidget, QTableWidgetItem, QDialog, QLabel, QHBoxLayout, QInputDialog, QCheckBox, QButtonGroup, QHBoxLayout, \
    QRadioButton, QScrollBar, QDoubleSpinBox, QSizePolicy, QSlider, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, QRectF, QSize
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QCursor
from eeg_main import Ui_MainWindow as ui_main
from sub_dialog import Ui_Dialog as ui_sub
from sub_rename import Ui_Dialog as ui_rename
from sub_reference import Ui_Dialog as ui_ref
from sub_multiplechoice import Ui_Dialog as ui_choice
from sub_drop import Ui_Dialog as ui_drop
from sub_reorder import Ui_Dialog as ui_reorder
from sub_eog import Ui_Dialog as ui_eog
from sub_split import Ui_Dialog as ui_split
from sub_check import Ui_Dialog as ui_check
from sub_video import Ui_Dialog as ui_video
from sub_sub_option import Ui_Dialog as ui_option
from sub_sub_event import Ui_Dialog as ui_event
from sub_sub_event_checker import Ui_Dialog as ui_event_checker
from sub_sync_movie import Ui_Dialog as ui_sync_movie
from sub_sync_self import Ui_Dialog as ui_sync_self
from sub_eegview import Ui_Dialog as ui_eegview_mpl
from sub_eegview_pyqtgraph import Ui_Dialog as ui_eegview_pyqtgraph
from sub_psi_settings import Ui_Dialog as ui_psi_settings
from sub_psi_viewer import Ui_Dialog as ui_psi_viewer
from sub_hyper_event_editor import Ui_Dialog as ui_hyper_event_editor
from sub_eegview_focus import Ui_Dialog as ui_eegview_focus
from sub_seed_target import Ui_Dialog as ui_seed_target
from sub_psi_statistical import Ui_Dialog as ui_psi_statistical
from sub_continuous_psi import Ui_Dialog as ui_continuous_psi
from sub_psi_load import Ui_Dialog as ui_load_psi
from sub_continuous_plotting import Ui_Dialog as ui_continuous_plotting
from sub_information_plot import Ui_Dialog as ui_information_plot
from sub_statistical_mi import Ui_Dialog as ui_statistical_mi
from sub_auto_peak_detect import Ui_Dialog as ui_auto_peak_detect
import os
import mne
from PyQt5.QtGui import QColor
import math
import pyqtgraph as pqg
import random
from mne.time_frequency.tfr import morlet as mne_morlet
from scipy.signal import convolve, gaussian, argrelmax, decimate
import seaborn as sns
from multiprocessing import Pool, Array, Value
import glob
import ctypes
from tqdm import tqdm
from numba import jit, f4, i4

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
import matplotlib.pyplot
import time

from mne.preprocessing import ICA
import numpy as np
from multiprocessing import Process, Queue
import multiprocessing
import subprocess
import pyautogui as pg
from time import sleep
from numpy.polynomial.legendre import legval
from scipy.linalg import inv


################################################################################################
################################################################################################
# Functions
################################################################################################
################################################################################################
def save_set(raw, fname):
    """
    save raw file as "set" file
    -------------------
    Parameters:
        raw: raw object
        fname: file path

    """
    from scipy.io import savemat
    from numpy.core.records import fromarrays
    data = raw.get_data() * 1e6  # microvolt
    fs = float(raw.info["sfreq"])
    times = raw.times
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])
    events = fromarrays([raw.annotations.description,
                         raw.annotations.onset * fs + 1,
                         raw.annotations.duration * fs],
                        names=["type", "latency", "duration"])
    savemat(fname, dict(EEG=dict(data=data,
                                 setname=fname,
                                 nbchan=data.shape[0],
                                 pnts=float(data.shape[1]),
                                 trials=1,
                                 srate=fs,
                                 xmin=times[0],
                                 xmax=times[-1],
                                 chanlocs=chanlocs,
                                 event=events,
                                 icawinv=[],
                                 icasphere=[],
                                 icaweights=[])),
            appendmat=False)


def adjust_MATLAB(raw, rename_chs, chs_num, ced_path, eog_chs, epoch_t=10.0, sfreq=500):

    import matlab.engine
    eng = matlab.engine.start_matlab()
    epoch_t = epoch_t
    i = 0
    while True:
        raw_Ac = raw.copy()
        try:
            raw_Ac.crop(i * epoch_t, (i + 1) * epoch_t)
        except:
            raw_adA.rename_channels(rename_chs)

            if eog_chs != []:
                raw_eog = raw.copy().pick_channels(eog_chs).crop(0, i * epoch_t - 1 / sfreq)
                raw_adA.add_channels([raw_eog], force_update_info=True)

            eng.quit()
            return raw_adA

        with open("./pass.txt", "w") as f:
            f.write(str(chs_num) + " ")
            f.write(ced_path)

        save_set(raw_Ac, "pass.set")

        try:
            eng.eeg_mne(nargout=0)

        except:
            print("undefined error")
            raw_o = mne.io.read_raw_eeglab("test_after.set", preload=True)
            raw_o.crop(0, epoch_t - 1 / sfreq)
            raw_adA.append(raw_o)
            i += 1
            continue

        with open("report.txt", "r") as f:
            lines = f.readlines()
            rep_v = lines[-2].replace("\n", "").split()
            rep_v = [int(x) for x in rep_v]
            # print(rep_v)
            # print(lines[-2])
            if rep_v == []:
                rep_v = [1000]
            np.savetxt("rep_a.csv", rep_v, delimiter=",")
        eng.eeg_mne2(nargout=0)

        if i >= 1:
            raw_o = mne.io.read_raw_eeglab("test_after.set", preload=True)
            raw_o.crop(0, epoch_t - 1 / sfreq)
            raw_adA.append(raw_o)
            i += 1

        if i == 0:
            raw_adA = mne.io.read_raw_eeglab("test_after.set", preload=True)

            raw_adA.crop(0, epoch_t - 1 / sfreq)
            i += 1


def ica_auto(raw, chunks_t=20.0, method="infomax", eogs=["A_ELL", "A_ERT", "A_ERU", "A_ERR"], sfreq=500):
    """
    Only "eog" components are removed from the raw object automatically by mne function
    -------------------
    Parameters:
        raw: raw object
        chunks_t: window time for applying ica
        method: ica method
        eogs: eog channels should be pre-selected. They are selected in App.
        sfreq: sampling frequency
    -------------------
    Returns:
        raw object applied eog-ica
    -------------------
    """
    from IPython.display import clear_output

    chunks_n = int(raw.get_data().shape[1] // (chunks_t * sfreq))
    for i in range(chunks_n):
        raw_tmp = raw.copy()
        raw_tmp.crop(tmin=chunks_t * i, tmax=chunks_t * (i + 1) - 1 / sfreq)
        ica = ICA(random_state=0, method=method)
        ica.fit(raw_tmp, reject_by_annotation=True)
        rejs = []
        for eog in eogs:
            rej, _ = ica.find_bads_eog(raw_tmp, ch_name=eog)
            rejs = rejs + rej
        rejs = list(set(rejs))
        ica.apply(raw_tmp, exclude=rejs)
        clear_output()
        if i == 0:
            raw_ret = raw_tmp.copy()
            # raw_ret.plot(n_channels=len(A_chs))
        if i >= 1:
            raw_ret.append(raw_tmp)
    return raw_ret


def csd_apply(raw_arr, ch_names, csd_fullpath, stiffnes, lambda2):
    """
    ** This function is almost from pyCSD.
    current source density is applied to raw object.
    -------------------
    Parameters:
        raw_arr: array from raw object
        ch_names: ch_names
        csd_fullpath: full path of csd file
        stiffnes, lambda2: parameters of csd
    -------------------
    Returns:
        array applied csd
    """
    def calc_g(cosang, stiffnes=4, num_lterms=50):
        factors = [(2 * n + 1) / (n ** stiffnes * (n + 1) ** stiffnes * 4 * np.pi)
                   for n in range(1, num_lterms + 1)]
        return legval(cosang, [0] + factors)

    def calc_h(cosang, stiffnes=4, num_lterms=50):
        factors = [(2 * n + 1) /
                   (n ** (stiffnes - 1) * (n + 1) ** (stiffnes - 1) * 4 * np.pi)
                   for n in range(1, num_lterms + 1)]
        return legval(cosang, [0] + factors)

    def prepare_G(G, lambda2):
        # regularize if desired
        if lambda2 is None:
            lambda2 = 1e-5

        G.flat[::len(G) + 1] += lambda2
        # compute the CSD
        Gi = inv(G)

        TC = Gi.sum(0)
        sgi = np.sum(TC)  # compute sum total

        return Gi, TC, sgi

    def compute_csd(data, G_precomputed, H, head):
        n_channels, n_times = data.shape
        mu = data.mean(0)[None]
        Z = data - mu
        X = np.zeros_like(data)
        head **= 2

        Gi, TC, sgi = G_precomputed

        Cp2 = np.dot(Gi, Z)
        c02 = np.sum(Cp2, axis=0) / sgi
        C2 = Cp2 - np.dot(TC[:, None], c02[None, :])
        X = np.dot(C2.T, H).T / head
        return X

    montage = mne.channels.read_montage(csd_fullpath)
    pos_picks = [montage.ch_names.index(x) for x in ch_names]
    pos = montage.pos[pos_picks]  # Cartesian space value

    cosang = np.dot(pos, pos.T)
    G = calc_g(cosang, stiffnes=stiffnes)
    H = calc_h(cosang, stiffnes=stiffnes)

    G_precomputed = prepare_G(G.copy(), lambda2=lambda2)

    X = compute_csd(raw_arr, G_precomputed=G_precomputed, H=H, head=1)

    return X


def merge_AB(raw_A, raw_B, chs_n, events, sfreq=500):
    """
    merge two raw objects (vertically). ch_types are changed to "eeg".
    -------------------
    Parameters:
        raw_A, raw_B: raw object
        chs_n: ch nums
        events: events
        sfreq: sampling frequency
    -------------------
    Returns:
        merged raw objects
    """
    raw_merge = np.vstack((raw_A[:][0], raw_B[:][0]))
    sfreq = sfreq
    info = mne.create_info(ch_names=chs_n, sfreq=sfreq, ch_types=["eeg"] * len(chs_n))
    raw = mne.io.RawArray(raw_merge, info)

    events_inv = {v: k for k, v in events[1].items()}
    annot = mne.Annotations(0, 0.1, "first")
    for event in events[0]:
        annot.append(event[0] / sfreq, 0.01, events_inv[event[2]])
    try:
        annot.delete(0)
        raw.set_annotations(annot)
    except:
        pass

    return raw


def psi_morlet_by_arr(raw_arr, seed, target, tmin_m, tmax_m, cwt_freqs, cwt_n_cycles, sfreq=500, faverage=True):
    """
    Psi calculation by arr.
    Morlet transform makes artifacts in both edges, so margins are needed.
    -------------------
    Parameters:
        raw_arr: array of raw objects. margins should be included.
        seed: seed index of raw_arr
        target: target index of raw_arr
        tmin_m: margin of before segment of calculation
        tmax_m: margin of after segment of calculation
        cwt_freqs: numpy array like np.array([8, 8.5, 9, 9.5 ...]). morlet frequencies for calculation of psi.
        cwt_n_cycles: n_cycles of wavelet. This app uses cwt_freqs / 2.
        sfreq: sampling frequency
        faverage: bool. if True, average is calculated along the frequency axis.
    -------------------
    Returns:
        psi array
    """
    seed_arr = raw_arr[seed, :]
    target_arr = raw_arr[target, :]
    data_p = raw_arr.shape[1]
    freq_n = len(cwt_freqs)

    morlets = mne_morlet(sfreq=sfreq, freqs=cwt_freqs, n_cycles=cwt_n_cycles)
    convs = np.empty((freq_n, 2, data_p), dtype="complex128")
    for m, morlet_ in enumerate(morlets):
        for i, raw_ch in enumerate([seed_arr, target_arr]):
            convs[m][i][:] = convolve(raw_ch, morlet_, mode="same")

    phase_arr = np.angle(convs)

    T_length = int(data_p - tmin_m * sfreq - tmax_m * sfreq)

    imag = 0j
    img_perf = []

    for f in range(freq_n):
        for t in range(int(tmin_m * sfreq), int(tmin_m * sfreq + T_length)):
            phase_diff = phase_arr[f, 0, t] - phase_arr[f, 1, t]
            phase_diff_euler = np.exp(1j * phase_diff)
            imag += phase_diff_euler
        img_perf.append(float(np.abs(imag) / T_length))
        imag = 0j

    if faverage:
        return sum(img_perf) / len(img_perf)
    else:
        return img_perf


def psi_morlet_by_arr_r(raw_arr, seed, target, tmin, tmax, cwt_freqs, cwt_n_cycles, sfreq=500, faverage=True):
    """
    This function is the version of not considering margin artifacts of psi_morlet_by_arr.
    """
    seed_arr = raw_arr[seed, :]
    target_arr = raw_arr[target, :]
    data_p = raw_arr.shape[1]
    freq_n = len(cwt_freqs)

    morlets = mne_morlet(sfreq=sfreq, freqs=cwt_freqs, n_cycles=cwt_n_cycles)
    convs = np.empty((freq_n, 2, data_p), dtype="complex128")
    for m, morlet_ in enumerate(morlets):
        for i, raw_ch in enumerate([seed_arr, target_arr]):
            convs[m][i][:] = convolve(raw_ch, morlet_, mode="same")

    phase_arr = np.angle(convs)

    T_length = int((tmax - tmin) * sfreq)

    imag = 0j
    img_perf = []

    for f in range(freq_n):
        for t in range(int(tmin * sfreq), int(tmin * sfreq + T_length)):
            phase_diff = phase_arr[f, 0, t] - phase_arr[f, 1, t]
            phase_diff_euler = np.exp(1j * phase_diff)
            imag += phase_diff_euler
        img_perf.append(float(np.abs(imag) / T_length))
        imag = 0j

    if faverage:
        return sum(img_perf) / len(img_perf)
    else:
        return img_perf


def psi_morlet_cont_arr(raw_arr, seed, target, tmin, tmax, window_size_t, stride_t, cwt_freqs, cwt_n_cycles,
                        sfreq=500, faverage=True):
    """
    calculate continuous psi array. Firstly, morlet wavelet transform is conducted to whole array,
    so need not to think about margin artifacts.
    -------------------
    Parameters:
        raw_arr: numpy array of raw objects
        seed: index of seed channel
        target: index of target channel
        tmin: calculation start time (s)
        tmax: calculation end time (s)
        window_size_t: window length (s) for applying psi calculation
        stride_t: For each calculation, window_size_t is shifted by stride_t (s)
        cwt_freqs: morlet frequencies for applying calculation. See psi_morlet_by_arr.
        cwt_n_cycles: This App uses cwt_freqs / 2
        sfreq: sampling frequency
        faverage: bool. If True, average is calculated along the frequency axis.
    -------------------
    Returns:
        psi array(freq num, window onset time num),  window onset time array
    """
    seed_arr = raw_arr[seed, :]
    target_arr = raw_arr[target, :]
    data_p = raw_arr.shape[1]
    freq_n = len(cwt_freqs)

    morlets = mne_morlet(sfreq=sfreq, freqs=cwt_freqs, n_cycles=cwt_n_cycles)
    convs = np.empty((freq_n, 2, data_p), dtype="complex128")

    for m, morlet_ in enumerate(morlets):
        for i, raw_ch in enumerate([seed_arr, target_arr]):
            convs[m][i][:] = convolve(raw_ch, morlet_, mode="same")
    phase_arr = np.angle(convs)

    Tmin = tmin * sfreq
    Tmax = tmax * sfreq

    Window_size = window_size_t * sfreq

    n_window = int((tmax - window_size_t - tmin) // stride_t)

    plv_cap = []
    window_onset_time = []

    imag = 0j
    img_perf = []

    for n in range(n_window):
        Tmin_ab = int(Tmin + n * stride_t * sfreq)
        Tmax_ab = int(Tmin_ab + Window_size)
        for f in range(freq_n):
            for t in range(Tmin_ab, Tmax_ab):
                phase_diff = phase_arr[f, 0, t] - phase_arr[f, 1, t]
                phase_diff_euler = np.exp(1j * phase_diff)
                imag += phase_diff_euler
            img_perf.append(float(np.abs(imag) / Window_size))
            imag = 0j

        if faverage:
            plv_cap.append(sum(img_perf) / len(img_perf))
        else:
            plv_cap.append(img_perf)
        window_onset_time.append(Tmin_ab / sfreq)
        img_perf = []

    return plv_cap, window_onset_time


def psi_calc(args):
    """
    This function is for parallel computing for psi calculation.
    """
    events_onset, raw, fr_range, seed_channels_ind, target_channels_ind, margin, window_t, step_fr, sfreq = args

    ret_arr = np.empty((len(fr_range), len(seed_channels_ind), len(target_channels_ind)))
    raw_tmp = raw.copy()
    raw_tmp.crop(events_onset - margin[0], events_onset + window_t + margin[1])
    raw_arr = raw_tmp.get_data()
    for l, (fmin, fmax) in enumerate(fr_range):
        cwt_freqs = np.arange(float(fmin), float(fmax), step_fr)
        for m, seed_ind in enumerate(seed_channels_ind):
            for n, target_ind in enumerate(target_channels_ind):
                print("({}, {})".format(m, n))
                psi_val = psi_morlet_by_arr(raw_arr, seed=seed_ind, target=target_ind, tmin_m=margin[0],
                                            tmax_m=margin[1], cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_freqs / 2,
                                            sfreq=sfreq, faverage=True)
                ret_arr[l, m, n] = psi_val
    return ret_arr


def permutation_test(args):
    """
    This function is for parallel computing for permutation test.
    """
    target_channels_ind, task_psi_ls, nontask_psi_ls, permutation_times, epoch_num = args

    ret_arr = np.empty(len(target_channels_ind))

    for t in range(len(target_channels_ind)):
        task_psis = task_psi_ls[t, :]
        nontask_psis = nontask_psi_ls[t, :]

        diff_true = (sum(task_psis) / len(task_psis)) - (sum(nontask_psis) / len(nontask_psis))
        base_array = np.hstack((task_psis, nontask_psis))
        diffs_sample = []
        for i in range(permutation_times):
            sample = np.random.choice(base_array, epoch_num * 2, replace=True)
            sample_f = sample[:epoch_num]
            sample_s = sample[epoch_num:]
            diffs_sample.append(np.mean(sample_f) - np.mean(sample_s))

        p_value = np.sum(diffs_sample > diff_true) / len(diffs_sample)
        ret_arr[t] = p_value
    return ret_arr


def plot_p(p, seed_channels, target_channels, fr_range, mode, fdr, dirpath=""):
    for i in range(p.shape[0]):
        fig, ax = matplotlib.pyplot.subplots(figsize=(p.shape[2] / 2, p.shape[1] / 2 + 1))
        sns.heatmap(p[i, :, :], annot=True, vmin=0, vmax=0.1, fmt='.2f',
                    yticklabels=seed_channels,
                    xticklabels=target_channels,
                    cmap="Reds_r", ax=ax)
        ax.set_ylim(p.shape[1], 0)
        if fdr:
            ax.set_title("{} - {} Hz, p-fdr".format(fr_range[i][0], fr_range[i][1]))
        else:
            ax.set_title("{} - {} Hz, p-values".format(fr_range[i][0], fr_range[i][1]))

        if mode == "save":
            matplotlib.pyplot.savefig(dirpath + "{}_{}.png".format(fr_range[i][0], fr_range[i][1]))
    if mode == "plot":
        matplotlib.pyplot.show()

@jit(f4[:, :](f4[:, :, :], i4, i4, f4, f4, i4))
def calc_psi_cap(phase_arr, freq_n, n_window, window_t, stride_t, sfreq):
    psi_cap = np.empty((freq_n, n_window), dtype=np.float32)
    imag = 0j
    window_size = window_t * sfreq

    for n in range(n_window):
        tmin_ab_ind = int(n * stride_t * sfreq)
        tmax_ab_ind = int(tmin_ab_ind + window_size)
        for f in range(freq_n):
            for t in range(tmin_ab_ind, tmax_ab_ind):
                phase_diff = phase_arr[f, 0, t] - phase_arr[f, 1, t]
                phase_diff_euler = np.exp(1j * phase_diff)
                imag += phase_diff_euler
            psi_cap[f, n] = float(np.abs(imag) / window_size)
            imag = 0j

    return psi_cap


def continuous_psi_async(args):
    """
    This function is for parallel computing for psi continuous calculation.
    """
    seed_arr, target_arr, seed_ind, target_ind, window_t, stride_t, cwt_freqs, cwt_n_cycles, sfreq = args

    data_p = seed_arr.shape[-1]
    morlets = mne_morlet(sfreq=sfreq, freqs=cwt_freqs, n_cycles=cwt_n_cycles)
    freq_n = len(cwt_freqs)
    convs = np.empty((freq_n, 2, data_p), dtype="complex64")

    for m, morlet in enumerate(morlets):
        for i, raw_ch in enumerate([seed_arr, target_arr]):
            convs[m][i][:] = convolve(raw_ch, morlet, mode="same")

    phase_arr = np.angle(convs)
    n_window = int((data_p / sfreq - window_t) // stride_t)

    psi_cap = calc_psi_cap(phase_arr, freq_n, n_window, window_t, stride_t, sfreq)

    return psi_cap


def bins_Freedman(data, tmin, tmax, sfreq=500):
    """
    Calculation of bins number for information theory by Freedman method.
    -------------------
    Parameters:
        data: data array
        tmin: start time (s)
        tmax: end time (s)
        sfreq: sampling frequency
    -------------------
    Returns:
        optimal bins number
    """
    optimal_bins = []
    for i in range(data.shape[0]):
        anal_array = data[i, int(tmin * sfreq): int(tmax * sfreq)]
        Q75, Q25 = np.percentile(anal_array, [75, 25])
        data_n = (tmax - tmin) * sfreq
        optimal_bin = np.ceil((np.max(anal_array) - np.min(anal_array)) * (data_n ** (1 / 3)) / (2 * (Q75 - Q25)))
        optimal_bins.append(optimal_bin)

    optimal_bin_num = np.ceil(np.mean(optimal_bins))
    return int(optimal_bin_num)


def entropy_calc(data, tmin, tmax, bins, base_subtraction=False, base_tmin=-0.4, base_tmax=-0.1, sfreq=500):
    """
    Information entropy is calculated by corrected base segment (True or False).
    -------------------
    Parameters:
        data: data array
        tmin: start time (s)
        tmax: end time (s)
        bins: bins for histogram
        base_subtraction: If True, base segment is used for correction
        base_tmin: relative base segment start time (s)
        base_tmax: relative base segment end time (s)
        sfreq: sampling frequency
    -------------------
    Returns:
        entropy array

    """
    entropies = []
    entropies_sub = []
    for i in range(data.shape[0]):
        hist, bin_ = np.histogram(data[i, int(tmin * sfreq): int(tmax * sfreq)], bins=bins)
        elesum = sum(hist)
        entro_elems = [(x / elesum) * np.log2(x / elesum + sys.float_info.epsilon) for x in hist]
        entropy = -1 * (sum(entro_elems))
        entropies.append(entropy)

        if base_subtraction == True:
            hist, bin_ = np.histogram(data[i, int(base_tmin * sfreq): int(base_tmax * sfreq)], bins=bins)
            elesum = sum(hist)
            entro_elems = [(x / elesum) * np.log2(x / elesum + sys.float_info.epsilon) for x in hist]
            entropy = -1 * (sum(entro_elems))
            entropies_sub.append(entropy)

    if base_subtraction == False:

        entro_res = np.array(entropies)

        return entro_res

    else:
        entro_res = np.array(entropies) - np.array(entropies_sub)

        return entro_res


def entropy_continuous_calc(data, tmin, tmax, window_t, stride_t, ch_indice, ch_names, plot_continuous=True, sfreq=200):
    """
    Continuous entropy plotting function
    -------------------
    Parameters:
        data: data array
        tmin: start time (s)
        tmax: end time (s)
        window_t: window length (s)
        stride_t: for each calculation, window_t is shifted by stride_t (s)
        ch_indice: channels indice for the calculation
        ch_names: all channel names for plotting
        plot_continuous: if True, plot the continuous entropy.
        sfreq: sampling frequency
    -------------------
    Returns:
        entropy continuous array, occuring time points (so you can plot as they are)
    """
    calc_num = int((tmax - window_t - tmin) // stride_t)

    bins = bins_Freedman(data, tmin, tmax, sfreq=sfreq)

    ch_indice_names = np.array(ch_names)[ch_indice]

    time_points = []

    for i in range(calc_num):
        time_points.append(tmin + i * stride_t)

    entropy_cont = np.empty((len(ch_indice), calc_num))
    for i in range(calc_num):
        entro = entropy_calc(data=data[ch_indice], tmin=tmin + i * stride_t, tmax=tmin + i * stride_t + window_t,
                             bins=bins, sfreq=sfreq)
        entropy_cont[:, i] = entro

    if plot_continuous:
        matplotlib.pyplot.figure(figsize=(20, 10))
        for i, ch_index in enumerate(ch_indice_names):
            matplotlib.pyplot.plot(time_points, entropy_cont[i, :], label=ch_index)

        matplotlib.pyplot.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=8, ncol=2)
        matplotlib.pyplot.title("entropy_overtime")
        matplotlib.pyplot.xlabel("time(s)")
        matplotlib.pyplot.ylabel("entropy(bits)")
        matplotlib.pyplot.show()
        print("finish")

    return entropy_cont, time_points


def joint_entropy_calc(data_2d, tmin, tmax, bins, sfreq=200):
    """
    Joint entropy calculation
    -------------------
    Parameters:
        data_2d: data array that have two raws
        tmin: start time (s)
        tmax: end time (s)
        bins: bins num for calculation
        sfreq: sampling frequency
    -------------------
    Returns:
        joint entropy value
    """
    import sys
    data_anal = data_2d[:, int(tmin * sfreq): int(tmax * sfreq)]
    H, _, _ = np.histogram2d(data_anal[0], data_anal[1], bins=bins)

    total = np.sum(H)
    H_calc = ((H + sys.float_info.epsilon) / total) * np.log2((H + sys.float_info.epsilon) / total)

    return -1 * np.sum(H_calc)


def mutual_information_calc(data_2d, tmin, tmax, bins, sfreq=200):
    """
    Mutual information entropy calculation
    -------------------
    Parameters:
        data_2d: data array that have two raws
        tmin: start time (s)
        tmax: end time (s)
        bins: bins num for calculation
        sfreq: sampling frequency
    -------------------
    Returns:
        mutual_infomation value
    """
    entropy0 = entropy_calc(data=data_2d[[0], :], tmin=tmin, tmax=tmax, bins=bins, sfreq=sfreq)
    entropy1 = entropy_calc(data=data_2d[[1], :], tmin=tmin, tmax=tmax, bins=bins, sfreq=sfreq)
    joint_entropy = joint_entropy_calc(data_2d, tmin=tmin, tmax=tmax, bins=bins, sfreq=sfreq)

    return float(entropy0 + entropy1 - joint_entropy)


def mutual_information_continuous(data_2d, tmin, tmax, window_t, stride_t, seed_ch_name, target_ch_name,
                                  plot_continuous=True, sfreq=200):
    """
    Mutual information is calculated one by one, and plot them.
    -------------------
    Parameters:
        data_2d: data array that have two raws
        tmin: start time (s)
        tmax: end time (s)
        window_t: window length (s)
        stride_t: For each calculation, window_t is shifted along the time axis.
        seed_ch_name: seed channel name (str)
        target_ch_name: target channel name (str)
        plot_continuous: If True, plot the values
        sfreq: sampling frequency
    -------------------
    Returns:
        mutual information values, time_points (so you can plot as they are)
    """
    calc_num = int((tmax - window_t - tmin) // stride_t)

    time_points = []
    MI_cont = []

    bins = bins_Freedman(data_2d, tmin, tmax, sfreq=sfreq)

    for i in range(calc_num):
        time_points.append(tmin + i * stride_t)

    for i in range(calc_num):
        MI = mutual_information_calc(data_2d=data_2d, tmin=tmin + i * stride_t, tmax=tmin + i * stride_t + window_t,
                                     bins=bins, sfreq=sfreq)
        MI_cont.append(MI)

    if plot_continuous == True:
        matplotlib.pyplot.figure(figsize=(20, 10))
        matplotlib.pyplot.plot(time_points, MI_cont)
        matplotlib.pyplot.title("{}-{} MI_overtime".format(seed_ch_name, target_ch_name))
        matplotlib.pyplot.ylabel("MI(bits)")
        matplotlib.pyplot.xlabel("time(s)")
        matplotlib.pyplot.show()

    return MI_cont, time_points


def lagged_mutual_information(data_2d, tmin, tmax, lagtime_window_t, lag_nums, seed_ch_name, target_ch_name, seed=0,
                              plot=True, sfreq=200):
    """
    Lagged mutual information entropies calculation
    -------------------
    Parameters:
        data_2d: data array that have two raws
        tmin: start time (s)
        tmax: end time (s)
        lagtime_window_t: window length (s)
        lag_nums: For each calculation, target array is shifted from the seed by range(lag_nums), and calculated.
        seed_ch_name: seed channel name (str)
        target_ch_name: target channel name (str)
        seed: seed index (0 or 1)
        plot: if True, plot the result
        sfreq: sampling frequency
    -------------------
    Returns:
        lagged mutual information values, time_points (so you can plot them as they are)

    """
    data_seed = data_2d[seed, int(tmin * sfreq): int(tmax * sfreq)]
    data_target = data_2d[1 - seed]

    MI_cont = []
    time_points = []

    bins = bins_Freedman(data_2d, tmin, tmax, sfreq=sfreq)

    for i in range(lag_nums):
        data_2d = np.vstack((data_seed, data_target[int((tmin + i * lagtime_window_t) * sfreq): int(
            (tmax + i * lagtime_window_t) * sfreq)]))
        MI = mutual_information_calc(data_2d, tmin=0, tmax=tmax - tmin, bins=bins, sfreq=sfreq)
        MI_cont.append(MI)
        time_points.append(i * lagtime_window_t)

    if plot:
        matplotlib.pyplot.figure(figsize=(20, 10))
        matplotlib.pyplot.plot(time_points, MI_cont)
        matplotlib.pyplot.title("{}-{} Lagged MI".format(seed_ch_name, target_ch_name))
        matplotlib.pyplot.xlabel("lag time(s)")
        matplotlib.pyplot.ylabel("MI")
        matplotlib.pyplot.show()

    return MI_cont, time_points


def permutation_mutual_information(args):
    """
    This function is for parallel computing of mutual information permutation test.
    """
    data_2d, data_t, seed_ons, target_ons, window_t, shift_minus, shift_plus, sampling_num, sfreq = args

    data_seed = data_2d[0, int(seed_ons * sfreq): int((seed_ons + window_t) * sfreq)]
    data_target = data_2d[1, int(seed_ons * sfreq): int((seed_ons + window_t) * sfreq)]

    bins = bins_Freedman(data_2d, seed_ons, seed_ons + window_t, sfreq=sfreq)

    data_2d_base = np.vstack((data_seed, data_target))
    MI_base = mutual_information_calc(data_2d_base, tmin=0, tmax=window_t, bins=bins, sfreq=sfreq)

    MI_tests = []
    for i in range(sampling_num):
        sel_tp = np.random.choice(
            list(range(int((target_ons + shift_minus) * sfreq), int((target_ons + shift_plus) * sfreq))), 1)
        data_2d_test = np.vstack((data_seed, data_t[int(sel_tp): int(sel_tp + window_t * sfreq)]))
        MI_test = mutual_information_calc(data_2d_test, tmin=0, tmax=window_t, bins=bins, sfreq=sfreq)
        MI_tests.append(MI_test)

    p_value = np.sum(np.array(MI_tests) > MI_base) / len(MI_tests)

    return p_value


def check_row(func):
    """
    Decorator of checking Mainwindow EEGFilesList. Check the currentRow and if none, set to 0.
    """

    def b(self):
        if self.ui.EEGFilesList.currentRow() == -1 and self.ui.EEGFilesList.count() > 0:
            self.ui.EEGFilesList.setCurrentRow(0)
        func(self)

    return b


def check_raw(func):
    """
    Decorator of checking Mainwindow EEGFilesList. Check the file name whether it includes "raw".
    """

    def b(self):
        tmp_txt = self.ui.EEGFilesList.currentItem().text()
        if not "raw" in tmp_txt:
            self.ui.TextBrowserComment.setText("Select the raw object")
        else:
            func(self)

    return b


def check_raws(func):
    """
    Decorator of checking Mainwindow raws list. Check the raws whether it includes raw object.
    """

    def b(self):
        if self.raws == []:
            self.ui.TextBrowserComment.setText("raw is empty")
        else:
            func(self)

    return b


def check_montage(func):
    """
    Decorator of checking Mainwindow. Check the montage file path exists or not.
    """

    def b(self):
        try:
            self.ui.TextBrowserMontageInfo.toPlainText()
        except:
            self.ui.TextBrowserComment.setText("Input the montage_path")

    return b


def check_list_nums(func):
    """
    Decorator of checking Mainwindow. Check whether raw num in raws is equal to EEGFilesList raws num.
    """

    def b(self):
        func(self)
        if len(self.raws) != self.ui.EEGFilesList.count():
            self.ui.TextBrowserComment.setText("Numbers don't match")

    return b

#######################################################################################################
#######################################################################################################
# MainWindow
#######################################################################################################
#######################################################################################################
class MyMainWindow(QMainWindow):
    """
    Main Window of this App.
    """
    def __init__(self):
        ######################
        super().__init__()
        self.ui = ui_main()
        self.ui.setupUi(self)
        ######################
        self.setWindowTitle("TomakunApp")

        # color
        self.ui.label_3.setStyleSheet("background-color: rgb(255,250,205)")
        self.ui.label_6.setStyleSheet("background-color: rgb(255,192,203)")
        # self.ui.centralwidget.setStyleSheet("background-color: rgb(240, 240, 240)")
        style = """
        QTabBar::tab:selected{background: rgba(212, 182, 255, 80)}
        QPushButton{background-color: rgb(219, 219, 219)}
        QTabWidget QPushButton{background-color: rgb(210, 210, 210)}
        QPushButton:hover{background-color: rgba(210, 210, 210, 80)}
        """
        self.setStyleSheet(style)

        # Buttons
        self.ui.PushButtonADDOrg.clicked.connect(self.addlist_eeg)
        self.ui.PushButtonAdd.clicked.connect(self.addlist_eeg)
        self.ui.PushButtonDelete.clicked.connect(self.deletelist_eeg)
        self.ui.PushButtonDeleteAll.clicked.connect(self.deletealllist_eeg)
        self.ui.PushButtonMontage.clicked.connect(self.montage_add)
        self.ui.PushButtonMontageAdd.clicked.connect(self.montage_add)
        self.ui.PushButtonDeleteMontage.clicked.connect(self.montage_delete)

        self.ui.PushButtonInput.clicked.connect(self.input_raws)

        self.ui.PushButtonRawPlot.clicked.connect(self.plot_raw)
        self.ui.PushButtonPlotPW.clicked.connect(self.plot_power)

        self.ui.PushButtonSaveFif.clicked.connect(self.save_as_fif)
        self.ui.PushButtonSaveSet.clicked.connect(self.save_as_set)

        self.ui.PushButtonAddMovie.clicked.connect(self.add_movie)
        self.ui.PushButtonDeleteMovie.clicked.connect(self.delete_movie)
        self.ui.PushButtonMovieLoad.clicked.connect(self.load_movie)

        self.ui.PushButtonOrig.clicked.connect(self.plot_raw_original)

        # Preprocessing
        self.ui.PushButtonRename.clicked.connect(self.rename)
        self.ui.PushButtonReorder.clicked.connect(self.reorder)
        self.ui.PushButtonNoREF.clicked.connect(self.set_ref)
        self.ui.PushButtonFiltering.clicked.connect(self.filtering)
        self.ui.PushButtonResampling.clicked.connect(self.resampling)
        self.ui.PushButtonDrop.clicked.connect(self.drop_channels)
        self.ui.PushButtonEOGO.clicked.connect(self.select_eog)
        self.ui.PushButtonSM.clicked.connect(self.set_montage)
        self.ui.PushButtonInterP.clicked.connect(self.interpolate_bad)
        self.ui.PushButtonICA.clicked.connect(self.run_ica)
        self.ui.PushButtonApplyICA.clicked.connect(self.apply_ica)

        # Analysis
        self.ui.PushButtonADDSecond.clicked.connect(self.addlist_eeg)
        self.ui.PushButtonMontageSecond.clicked.connect(self.montage_add)
        self.ui.PushButtonInputSecond.clicked.connect(self.input_raws)
        self.ui.PushButtonDropSecond.clicked.connect(self.drop_channels)
        self.ui.PushButtonSwapAB.clicked.connect(self.swap_ab)
        self.ui.PushButtonConcatenate.clicked.connect(self.concatenate_data)
        self.ui.PushButtonPSIStatistical.clicked.connect(self.psi_statistical)
        self.ui.PushButtonPSIContinuous.clicked.connect(self.psi_continuous)
        self.ui.PushButtonLoadPSI.clicked.connect(self.psi_load)
        self.ui.PushButtonStatisticalMutualInformation.clicked.connect(self.statistical_mi)

        # Hyperscanning original
        self.ui.PushButtonSelectEOG.clicked.connect(self.select_eog)
        self.ui.PushButtonSplit.clicked.connect(self.split_data)
        self.ui.PushButtonSMHP.clicked.connect(self.set_montagehp)
        self.ui.PushButtonInterPHP.clicked.connect(self.interpolate_bad)
        self.ui.PushButtonAdjust.clicked.connect(self.adjust_matlab)
        self.ui.PushButtonEOGCorrect.clicked.connect(self.eog_correct)
        self.ui.PushButtonCSD.clicked.connect(self.csd_apply)
        self.ui.PushButtonMerge.clicked.connect(self.merge_data)

        # List Widgets
        self.ui.EEGFilesList.itemClicked.connect(self.show_info)
        # Table Widgets
        self.ui.InfoTable.cellDoubleClicked.connect(self.show_detail_info)
        ######################
        # Menu Bars
        self.ui.actionOpen.triggered.connect(self.addlist_eeg)
        self.ui.actionSave_fif.triggered.connect(self.save_as_fif)
        self.ui.actionSave_set.triggered.connect(self.save_as_set)

        ######################
        # Attributes
        self.raws = []
        self.eog_channels = []
        self.icas = []
        self.video_dur = 0

        self.check = False

    @check_raws
    def save_as_fif(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory to save',
                                                    os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop")
        header, ok = QInputDialog.getText(self, 'If the names are the same, they will be overwritten', 'Enter the suffix name:', text="_preprocessed")
        if ok:
            for i, raw in enumerate(self.raws):
                raw.save("{}/{}{}.fif".format(dir_path, self.ui.EEGFilesList.item(i).text(), header), overwrite=True)

    @check_raws
    def save_as_set(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory to save',
                                                    os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop")
        header, ok = QInputDialog.getText(self, 'If the names are the same, they will be overwritten', 'Enter the suffix name:', text="_preprocessed")
        if ok:
            for i, raw in enumerate(self.raws):
                save_set(raw, "{}/{}{}.set".format(dir_path, self.ui.EEGFilesList.item(i).text(), header),
                         overwrite=True)

    def add_movie(self):
        file_name = QFileDialog.getOpenFileName(self, "load the movie file",
                                                os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop",
                                                "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.wmv)")[0]
        self.ui.TextBrowserMovie.setText(file_name)

    def delete_movie(self):
        self.ui.TextBrowserMovie.setText("")

    def load_movie(self):
        file_name = self.ui.TextBrowserMovie.toPlainText()

        if file_name != "":
            self.sub = VideoWindow(file_name, self)
            self.sub.show()

    def addlist_eeg(self):
        file_names = QFileDialog.getOpenFileNames(self, "open the EEG files",
                                                  os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop",
                                                  "vhdr or fif files (*.vhdr *.fif)")[0]
        for file in file_names:
            self.ui.EEGFilesList.addItem(file)

        self.ui.EEGFilesList.setCurrentRow(0)

        self.ui.TextBrowserComment.setText("File is added(not yet transformed to 'raw' object)")

    @check_row
    def deletelist_eeg(self):
        self.ui.EEGFilesList.takeItem(self.ui.EEGFilesList.currentRow())
        if len(self.raws) != 0:
            del self.raws[self.ui.EEGFilesList.currentRow()]

    def deletealllist_eeg(self):
        self.check = False
        sub = SubCheck(self, "Confirmation: Can I initialize?")
        sub.show()

        if self.check:
            self.ui.EEGFilesList.clear()
            self.raws = []
            self.eog_channels = []
            self.icas = []

            self.ui.PushButtonDelete.setEnabled(True)
            self.ui.PushButtonAdd.setEnabled(True)
            self.ui.PushButtonADDOrg.setEnabled(True)
            self.ui.PushButtonADDSecond.setEnabled(True)
            self.ui.TextBrowserComment.setText("Initialized")
        else:
            pass

    def montage_add(self):
        file_name = QFileDialog.getOpenFileName(self, "open the Montage file",
                                                os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop",
                                                "montage file (*.bvef)")[0]

        self.ui.TextBrowserMontageInfo.setText(file_name)

    def montage_delete(self):
        self.ui.TextBrowserMontageInfo.clear()

    def input_raws(self):
        cnts = self.ui.EEGFilesList.count()
        files = []
        for i in range(cnts):
            files.append(self.ui.EEGFilesList.item(i).text())
        print(files)

        for i, file in enumerate(files):
            if file[-5:] == ".vhdr":
                if i == 0:
                    self.ui.EEGFilesList.clear()

                i += 1
                raw = mne.io.read_raw_brainvision(file, preload=True)
                self.raws.append(raw)
                self.ui.EEGFilesList.addItem("raw_{}".format(os.path.splitext(os.path.basename(file))[0]))

            elif file[-4:] == ".fif":
                if i == 0:
                    self.ui.EEGFilesList.clear()

                i += 1
                raw = mne.io.read_raw_fif(file, preload=True)
                self.raws.append(raw)
                self.ui.EEGFilesList.addItem("raw_{}".format(os.path.splitext(os.path.basename(file))[0]))

            else:
                self.ui.TextBrowserComment.setText("This is not yet supported for other objects or other format files.")

        self.ui.EEGFilesList.setCurrentRow(0)
        self.ui.PushButtonDelete.setEnabled(False)
        # self.ui.PushButtonDeleteAll.setEnabled(False)
        self.ui.PushButtonAdd.setEnabled(False)
        self.ui.PushButtonADDOrg.setEnabled(False)
        self.ui.PushButtonADDSecond.setEnabled(False)
        self.ui.TextBrowserComment.setText("")

    @check_row
    @check_raw
    def plot_raw(self):
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        raw.plot(n_channels=len(raw.ch_names))

    @check_row
    @check_raw
    def plot_raw_original(self):
        ind_now = self.ui.EEGFilesList.currentRow()
        self.subview = SubViewerR(self, ind_now)
        self.subview.show()

    @check_row
    @check_raw
    def plot_power(self):
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        raw.plot_psd(fmin=0, fmax=80)

    def show_info(self):
        tmp_txt = self.ui.EEGFilesList.currentItem().text()
        if not "raw" in tmp_txt:
            tmp_item = QTableWidgetItem("file_path")
            self.ui.InfoTable.setItem(0, 0, tmp_item)
        else:
            ########################################################
            # Information Table
            tmp = QTableWidgetItem(tmp_txt)
            tmp.setBackground(QColor(192, 91, 118))
            self.ui.InfoTable.setItem(0, 0, tmp)

            tmp = QTableWidgetItem("Channels_num")
            tmp.setBackground(QColor(200, 200, 200))

            self.ui.InfoTable.setItem(1, 0, tmp)

            tmp = QTableWidgetItem("Channel_names")
            tmp.setBackground(QColor(225, 225, 225))

            self.ui.InfoTable.setItem(2, 0, tmp)

            tmp = QTableWidgetItem("sfreq")
            tmp.setBackground(QColor(200, 200, 200))

            self.ui.InfoTable.setItem(3, 0, tmp)

            tmp = QTableWidgetItem("Time_points(points)")
            tmp.setBackground(QColor(225, 225, 225))

            self.ui.InfoTable.setItem(4, 0, tmp)

            tmp = QTableWidgetItem("Time_points(seconds)")
            tmp.setBackground(QColor(200, 200, 200))

            self.ui.InfoTable.setItem(5, 0, tmp)

            tmp = QTableWidgetItem("Events")
            tmp.setBackground(QColor(225, 225, 225))

            self.ui.InfoTable.setItem(6, 0, tmp)

            self.ui.EEGFilesList.currentRow()

            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names
            ch_nums = len(ch_names)
            sfreq = raw.info["sfreq"]
            time_points = raw.n_times
            time_seconds = raw.n_times / sfreq

            events = mne.events_from_annotations(raw)

            self.ui.InfoTable.setItem(1, 1, QTableWidgetItem(" " + str(ch_nums)))
            self.ui.InfoTable.setItem(2, 1, QTableWidgetItem(" " + ",".join(ch_names)))
            self.ui.InfoTable.setItem(3, 1, QTableWidgetItem(" " + str(sfreq) + "Hz"))
            self.ui.InfoTable.setItem(4, 1, QTableWidgetItem(" " + str(time_points)))
            self.ui.InfoTable.setItem(5, 1, QTableWidgetItem(" " + str(time_seconds) + "s"))
            self.ui.InfoTable.setItem(6, 1, QTableWidgetItem(" " + str(len(events[0]))))

    def show_detail_info(self):

        tmp = self.ui.InfoTable.currentItem().text()
        if tmp and "raw" in self.ui.EEGFilesList.currentItem().text():
            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names
            events = mne.events_from_annotations(raw)
            text = "ch_names: {}".format(ch_names) + "\n\n" + "events: {}".format(events)

            sub = SubInfo(text)
            sub.show()
        else:
            pass

    @check_row
    def rename(self):
        if "raw" in self.ui.EEGFilesList.currentItem().text():
            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names

            self.ch_names_after = ""

            sub = SubRename(self, ",".join(ch_names))
            sub.show()

            # print(self.ch_names_after)
            rename_list = self.ch_names_after.split(",")
            changechs_dic = {x: y for x, y in zip(ch_names, rename_list)}

            try:
                for raw in self.raws:
                    raw.rename_channels(changechs_dic)
            except Exception as e:
                print(e.args)
                self.ui.TextBrowserComment.setText(e.args[0])

        else:
            pass

    @check_row
    def reorder(self):
        if "raw" in self.ui.EEGFilesList.currentItem().text():
            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names

            self.ch_names_after = ""

            sub = SubReorder(self, ",".join(ch_names))
            sub.show()

            # print(self.ch_names_after)
            reorder_list = self.ch_names_after.split(",")

            try:
                for raw in self.raws:
                    raw.reorder_channels(reorder_list)
            except Exception as e:
                print(e.args)
                self.ui.TextBrowserComment.setText(e.args[0])

        else:
            pass

    @check_raws
    @check_row
    def set_ref(self):
        eeg_lists = []
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        ch_names = raw.ch_names
        self.chosen = ""
        self.chosen_electrodes = []

        sub = SubRef(self, ",".join(ch_names))
        sub.show()

        if self.chosen == "N":
            for i, raw in enumerate(self.raws):
                eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_N")
                raw.set_eeg_reference(ref_channels=[])

        elif self.chosen == "A":
            for i, raw in enumerate(self.raws):
                eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_A")
                raw.set_eeg_reference(ref_channels="average")

        elif self.chosen == "M":
            for i, raw in enumerate(self.raws):
                eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_M")
                raw.set_eeg_reference(ref_channels=self.chosen_electrodes)

        self.ui.EEGFilesList.clear()
        self.ui.EEGFilesList.addItems(eeg_lists)
        self.ui.TextBrowserComment.setText("Ref Finished")

    @check_raws
    def filtering(self):
        lowpass_hz, ok = QInputDialog.getDouble(self, "low pass filter Hz", "low pass filter Hz", 1.0)
        highpass_hz, ok2 = QInputDialog.getDouble(self, "high pass filter Hz", "high pass filter Hz", 50.0,
                                                  lowpass_hz)

        if ok and ok2:
            eeg_lists = []

            for i, raw in enumerate(self.raws):
                eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_F")
                raw.filter(lowpass_hz, highpass_hz)

            self.ui.EEGFilesList.clear()
            self.ui.EEGFilesList.addItems(eeg_lists)
            self.ui.TextBrowserComment.setText("Filtering Finished")

        else:
            self.ui.TextBrowserComment.setText("Invalid")

    @check_raws
    def resampling(self):
        resample_rate, ok = QInputDialog.getDouble(self, "resampling rate", "resampling rate Hz", 500.0)

        if ok:
            eeg_lists = []

            for i, raw in enumerate(self.raws):
                eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_R")
                raw.resample(resample_rate)

            self.ui.EEGFilesList.clear()
            self.ui.EEGFilesList.addItems(eeg_lists)
            self.ui.TextBrowserComment.setText("Resampling Finished")

        else:
            self.ui.TextBrowserComment.setText("Invalid")

    @check_raws
    @check_row
    def drop_channels(self):
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        ch_names = raw.ch_names

        self.checked_channels = []

        sub = SubDrop(self, ch_names)
        sub.show()

        for i, raw in enumerate(self.raws):
            raw.drop_channels(self.checked_channels)

        self.ui.TextBrowserComment.setText(
            "Channel Drop Finished. Drop channels were {}".format(self.checked_channels))

    @check_row
    def run_ica(self):
        self.icas = []
        for raw in self.raws:
            ica = ICA(random_state=0)
            ica.fit(raw, reject_by_annotation=True)
            ica.plot_sources(raw)
            ica.plot_components(inst=raw)
            self.icas.append(ica)
        self.ui.TextBrowserComment.setText("ICA run. Choose the components and next apply them !")

    @check_raws
    @check_row
    def apply_ica(self):
        for i, raw in enumerate(self.raws):
            self.icas[i].apply(raw)
        self.ui.TextBrowserComment.setText("ICA applied !")

    @check_raws
    @check_row
    def select_eog(self):
        """
        This function cannot work when "A_" "B_" prefix doesn't exist
        """
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        ch_names = raw.ch_names

        ###############################################
        ch_names = [x.strip("A_") for x in ch_names]
        ch_names = [x.strip("B_") for x in ch_names]
        ###############################################

        self.eog_channels = []

        sub = SubEOG(self, ch_names)
        sub.show()
        print(self.eog_channels)

        # dic_tmp = {x: "eog" for x in self.eog_channels}

        for i, raw in enumerate(self.raws):
            if "A_" in self.ui.EEGFilesList.item(i).text():
                dic_tmp = {"A_" + x: "eog" for x in self.eog_channels}
                raw.set_channel_types(mapping=dic_tmp)
            elif "B_" in self.ui.EEGFilesList.item(i).text():
                dic_tmp = {"B_" + x: "eog" for x in self.eog_channels}
                raw.set_channel_types(mapping=dic_tmp)
            else:
                dic_tmp = {x: "eog" for x in self.eog_channels}
                raw.set_channel_types(mapping=dic_tmp)

        self.ui.TextBrowserComment.setText(
            "EOG select Finished. EOG channels were 'A_' or 'B_' or '' + {}".format(self.eog_channels))

    @check_raws
    @check_row
    def split_data(self):
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        ch_names = raw.ch_names

        self.split_channelsA = []
        self.split_channelsB = []

        sub = SubSplit(self, ",".join(ch_names))
        sub.show()

        print(self.split_channelsA)
        print(self.split_channelsB)

        try:
            eeg_lists = []
            raws_tmp = []

            for i, raw in enumerate(self.raws):
                eeg_lists.append("A_" + self.ui.EEGFilesList.item(i).text())
                eeg_lists.append("B_" + self.ui.EEGFilesList.item(i).text())
                raw_c = raw.copy()

                raw_A = raw.pick_channels(self.split_channelsA.split(","))
                raw_B = raw_c.pick_channels(self.split_channelsB.split(","))

                raws_tmp.append(raw_A)
                raws_tmp.append(raw_B)

            self.ui.EEGFilesList.clear()
            self.ui.EEGFilesList.addItems(eeg_lists)
            self.raws = raws_tmp
            self.ui.TextBrowserComment.setText("Split Finished")
        except:
            self.ui.TextBrowserComment.setText("failed")

    @check_raws
    @check_row
    def merge_data(self):
        raws_after = []
        eeg_lists = []
        raw = self.raws[self.ui.EEGFilesList.currentRow()]
        ch_names = raw.ch_names
        ################################################
        ch_names = [x.strip("A_") for x in ch_names]
        ch_names = [x.strip("B_") for x in ch_names]
        ################################################
        sfreq = raw.info["sfreq"]
        chs_n = ["A_" + x for x in ch_names] + ["B_" + x for x in ch_names]

        for i, raw in enumerate(self.raws):
            if i % 2 == 1:
                raws_after.append(
                    merge_AB(self.raws[i - 1], self.raws[i], chs_n=chs_n, events=mne.events_from_annotations(raw),
                             sfreq=sfreq))
                eeg_lists.append("Merged_" + self.ui.EEGFilesList.item(i - 1).text().strip("A_").strip("B_"))

        self.raws = raws_after
        self.ui.EEGFilesList.clear()
        self.ui.EEGFilesList.addItems(eeg_lists)
        self.ui.TextBrowserComment.setText("Merge success !")

    def set_montage(self):
        montage_path = self.ui.TextBrowserMontageInfo.toPlainText()

        try:
            montage = mne.channels.read_montage(montage_path)

            if self.ui.EEGFilesList.currentRow() == -1 and self.ui.EEGFilesList.count() > 0:
                self.ui.EEGFilesList.setCurrentRow(0)

            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names

            montage.ch_names = ['GND', 'REF', 'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
                                'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                                'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2',
                                'F4', 'F8', 'Fp2']

            mne.viz.plot_montage(montage)

            for raw in self.raws:
                raw.set_montage(montage)

        except Exception as e:
            print(e.args)
            self.ui.TextBrowserComment.setText(e.args[0])

    def set_montagehp(self):
        """
        This cannot work when other channels other than below exist.
        """
        montage_path = self.ui.TextBrowserMontageInfo.toPlainText()

        try:
            montageA = mne.channels.read_montage(montage_path)
            montageB = mne.channels.read_montage(montage_path)

            montageA.ch_names = ['GND', 'REF', 'A_Fp1', 'A_Fz', 'A_F3', 'A_F7', 'A_FT9', 'A_FC5', 'A_FC1', 'A_C3',
                                 'A_T7', 'A_TP9',
                                 'A_CP5', 'A_CP1', 'A_Pz', 'A_P3', 'A_P7', 'A_O1', 'A_Oz', 'A_O2', 'A_P4', 'A_P8',
                                 'A_TP10', 'A_CP6', 'A_CP2', 'A_Cz', 'A_C4', 'A_T8', 'A_FT10', 'A_FC6', 'A_FC2',
                                 'A_F4', 'A_F8', 'A_Fp2']

            montageB.ch_names = ['GND', 'REF', 'B_Fp1', 'B_Fz', 'B_F3', 'B_F7', 'B_FT9', 'B_FC5', 'B_FC1', 'B_C3',
                                 'B_T7', 'B_TP9',
                                 'B_CP5', 'B_CP1', 'B_Pz', 'B_P3', 'B_P7', 'B_O1', 'B_Oz', 'B_O2', 'B_P4', 'B_P8',
                                 'B_TP10', 'B_CP6', 'B_CP2', 'B_Cz', 'B_C4', 'B_T8', 'B_FT10', 'B_FC6', 'B_FC2',
                                 'B_F4', 'B_F8', 'B_Fp2']

            mne.viz.plot_montage(montageA)
            mne.viz.plot_montage(montageB)

            for i, raw in enumerate(self.raws):
                if "A_" in self.ui.EEGFilesList.item(i).text():
                    raw.set_montage(montageA)
                elif "B_" in self.ui.EEGFilesList.item(i).text():
                    raw.set_montage(montageB)
            self.ui.TextBrowserComment.setText("Montage set Finished")

        except Exception as e:
            print(e.args)
            self.ui.TextBrowserComment.setText(e.args[0])

    @check_raws
    def interpolate_bad(self):
        """
        This function works only "bad" channels
        """
        for raw in self.raws:
            raw.interpolate_bads()

    @check_row
    @check_raws
    @check_list_nums
    def adjust_matlab(self):

        pass_num, ok = QInputDialog.getInt(self, "note",
                                           "How many channels to apply ICA?\nNote: At this point, the channels must be sorted in the order of the ced files, and if they are EOG channels, they must be put together at the end.\nPlease select EOG again first when loading data.",
                                           26)
        adjust_t, ok2 = QInputDialog.getDouble(self, "note",
                                               "Enter the epoch time to apply the Adjust algorithm.", 10.0, 10.0)

        if ok and ok2:
            ced_path = QFileDialog.getOpenFileName(self, "select the ced file",
                                                   os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop",
                                                   "ced file (*.ced)")[0]


            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names
            sfreq = raw.info["sfreq"]

            ###############################################

            ch_names = [x.strip("A_") for x in ch_names]
            ch_names = [x.strip("B_") for x in ch_names]
            ###############################################
            ch_names_reorder_A = ["A_" + x for x in ch_names]
            ch_names_reorder_B = ["B_" + x for x in ch_names]

            rename_dic_A = {x: y for x, y in zip(ch_names[:pass_num], ch_names_reorder_A[:pass_num])}
            rename_dic_B = {x: y for x, y in zip(ch_names[:pass_num], ch_names_reorder_B[:pass_num])}

            eog_chs_A = ["A_" + x for x in self.eog_channels]
            eog_chs_B = ["B_" + x for x in self.eog_channels]

            cmd = "python auto_click_adjust.py"
            p = subprocess.Popen(cmd)

            raws_processed = []
            eeg_lists = []

            for i, raw in enumerate(self.raws):
                if "A_" in self.ui.EEGFilesList.item(i).text():
                    raws_processed.append(
                        adjust_MATLAB(raw, rename_dic_A, pass_num, ced_path, eog_chs_A, adjust_t, sfreq))
                    eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_J")

                elif "B_" in self.ui.EEGFilesList.item(i).text():
                    raws_processed.append(
                        adjust_MATLAB(raw, rename_dic_B, pass_num, ced_path, eog_chs_B, adjust_t, sfreq))
                    eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_J")

            self.raws = raws_processed
            self.ui.EEGFilesList.clear()
            self.ui.EEGFilesList.addItems(eeg_lists)

            p.kill()
            self.ui.TextBrowserComment.setText("Adjust ICA is finished.")


        else:
            self.ui.TextBrowserComment.setText("Processing was interrupted.")

    @check_row
    @check_raws
    @check_list_nums
    def eog_correct(self):
        chunks_t, ok = QInputDialog.getDouble(self, "note", "How many seconds do you apply ICA per segment?", 20.0, 10.0)
        if ok:
            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            sfreq = raw.info["sfreq"]

            eog_chs_A = ["A_" + x for x in self.eog_channels]
            eog_chs_B = ["B_" + x for x in self.eog_channels]

            raws_processed = []
            eeg_lists = []

            for i, raw in enumerate(self.raws):
                if "A_" in self.ui.EEGFilesList.item(i).text():
                    raws_processed.append(ica_auto(raw, chunks_t=chunks_t, eogs=eog_chs_A, sfreq=sfreq))
                    eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_E")

                elif "B_" in self.ui.EEGFilesList.item(i).text():
                    raws_processed.append(ica_auto(raw, chunks_t=chunks_t, eogs=eog_chs_B, sfreq=sfreq))
                    eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_E")

            self.raws = raws_processed
            self.ui.EEGFilesList.clear()
            self.ui.EEGFilesList.addItems(eeg_lists)
            self.ui.TextBrowserComment.setText("EOG Correct was finished.")

    @check_row
    @check_raws
    @check_list_nums
    def csd_apply(self):
        try:
            csd_path = QFileDialog.getOpenFileName(self, "select the csd file",
                                                   os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop",
                                                   "csd file (*.csd)")[0]

            raw = self.raws[self.ui.EEGFilesList.currentRow()]
            ch_names = raw.ch_names
            sfreq = raw.info["sfreq"]


            ch_names = [x.strip("A_") for x in ch_names]
            ch_names = [x.strip("B_") for x in ch_names]

            eog_idx = []

            for eog in self.eog_channels:
                eog_idx.append(ch_names.index(eog))


            noneog_idx = list(filter(lambda x: x not in eog_idx, range(len(ch_names))))


            ch_names_csd = []
            for idx in noneog_idx:
                ch_names_csd.append(ch_names[idx])


            raws_processed = []
            eeg_lists = []

            for i, raw in enumerate(self.raws):
                raw_b = raw.get_data()[noneog_idx, :]
                raw_e = raw.get_data()[eog_idx, :]

                raw_b = csd_apply(raw_b, ch_names=ch_names_csd, csd_fullpath=csd_path, stiffnes=4,
                                  lambda2=1e-5)

                raw_arcsd = np.concatenate([raw_b, raw_e], axis=0)

                ch_names_a = []

                if "A_" in self.ui.EEGFilesList.item(i).text():
                    ch_names_a = ["A_" + x for x in ch_names]

                elif "B_" in self.ui.EEGFilesList.item(i).text():
                    ch_names_a = ["B_" + x for x in ch_names]

                ch_types = ["eeg"] * len(ch_names)
                for idx in eog_idx:
                    ch_types[idx] = "eog"

                info = mne.create_info(ch_names_a, sfreq=sfreq, ch_types=ch_types)

                raw_after = mne.io.RawArray(raw_arcsd, info=info)

                ####################################################
                events = mne.events_from_annotations(raw)
                events_inv = {v: k for k, v in events[1].items()}
                annot = mne.Annotations(0, 0.1, "first")
                for event in events[0]:
                    annot.append(event[0] / sfreq, 0.01, events_inv[event[2]])
                try:
                    annot.delete(0)
                    raw_after.set_annotations(annot)
                except:
                    pass
                ###################################################

                raws_processed.append(raw_after)
                eeg_lists.append(self.ui.EEGFilesList.item(i).text() + "_C")

            self.raws = raws_processed
            self.ui.EEGFilesList.clear()
            self.ui.EEGFilesList.addItems(eeg_lists)
            self.ui.TextBrowserComment.setText("csd was finished.")
        except Exception as e:
            print(e.args)
            self.ui.TextBrowserComment.setText(e.args[0])

    @check_raws
    def concatenate_data(self):
        if self.ui.TextBrowserMovie.toPlainText() != "":
            sub = SubSyncMovie(self)
            sub.show()

        else:
            sub = SubSyncSelf(self)
            sub.show()

    @check_row
    @check_raw
    def swap_ab(self):
        raw = self.raws[self.ui.EEGFilesList.currentRow()]

        self.check = False
        sub = SubCheck(self, "Confirmation：Is it OK to reverse the channel with 'A_' and the channel with 'B_' (the order of channels should be the same)?")
        sub.show()

        if self.check:
            ch_names = raw.ch_names
            idx_start_a = [i for i, x in enumerate(ch_names) if x.startswith("A_")]
            idx_start_b = [i for i, x in enumerate(ch_names) if x.startswith("B_")]

            if len(idx_start_a) != len(idx_start_b) or len(idx_start_a) == 0:
                self.ui.TextBrowserComment.setText("Not possible due to wrong number of channels or bad name")
            else:
                annot_ons, annot_dur, annot_des = raw.annotations.onset, raw.annotations.duration, raw.annotations.description
                sfreq = raw.info["sfreq"]
                raw_arr = raw.get_data()

                arr_tmp = raw_arr[idx_start_b, :].copy()
                raw_arr[idx_start_b, :] = raw_arr[idx_start_a, :].copy()
                raw_arr[idx_start_a, :] = arr_tmp

                # for idx_a, idx_b in zip(idx_start_a, idx_start_b):
                #     raw_arr[idx_a, :], raw_arr[idx_b, :] = raw_arr[idx_b, :], raw_arr[idx_a, :]
                info = mne.create_info(ch_names, sfreq, ch_types=["eeg"] * len(ch_names))
                new_raw = mne.io.RawArray(raw_arr, info=info)
                new_raw.annotations.onset, new_raw.annotations.duration, new_raw.annotations.description = annot_ons, annot_dur, annot_des

                self.raws[self.ui.EEGFilesList.currentRow()] = new_raw
                self.ui.EEGFilesList.currentItem().setText(self.ui.EEGFilesList.currentItem().text() + "_SWAPPEDAB")

                self.ui.TextBrowserComment.setText("'A_' and 'B_' was swapped.")

    @check_row
    @check_raw
    def psi_statistical(self):
        sub = SubPSIStatistic(self, self.ui.EEGFilesList.currentRow())
        sub.show()

    @check_row
    @check_raw
    def psi_continuous(self):
        sub = SubPSIContinuous(self, self.ui.EEGFilesList.currentRow())
        sub.show()

    @check_row
    @check_raw
    def psi_load(self):
        self.choiced = 0
        self.choiced_cont_dir = ""

        sub = SubPSILoad(self)
        sub.show()

        if self.choiced == 1:
            ###############
            # continuous plotting mode
            self.continuous_plot_psi()
        #
        # elif self.choiced ==2:
        #     # statistical analysis mode
        #     self.statistical_analyze_psi()

    def continuous_plot_psi(self):
        self.sub_contp = SubContinuousPlotting(self, self.ui.EEGFilesList.currentRow())
        self.sub_contp.show()

    # def statistical_analyze_psi(self):
    #     self.sub_conts = SubLoadStatisticalAnalysis(self, self.ui.EEGFilesList.currentRow())
    #     self.sub_conts.show()

    @check_row
    @check_raw
    def statistical_mi(self):
        sub = StatisticalMI(self, self.ui.EEGFilesList.currentRow())
        sub.show()


##################################################################################################
##################################################################################################
# SubWindow
##################################################################################################
##################################################################################################
class StatisticalMI(QDialog):
    """
    Statistical mutual information entropy.
    """
    def __init__(self, parent, ind):
        super().__init__()
        self.ui = ui_statistical_mi()
        self.ui.setupUi(self)
        self.setWindowTitle("Statistical MI")
        self.parent = parent

        self.raw_seed = self.parent.raws[ind]

        self.seed_evt_onset = self.raw_seed.annotations.onset.tolist()
        self.seed_evt_description = self.raw_seed.annotations.description.tolist()

        #####################
        # scroll seed
        base = QWidget()
        base.setStyleSheet("background-color: honeydew")
        layout_seed = QVBoxLayout(base)
        self.rbs_seed = []  # checkbox

        for des in sorted(set(self.seed_evt_description), key=self.seed_evt_description.index):
            rb = QRadioButton(des)
            self.rbs_seed.append(rb)
            layout_seed.addWidget(rb)

        self.ui.ScrollAreaSeed.setWidget(base)
        #####################
        # target combo box
        for i in range(self.parent.ui.EEGFilesList.count()):
            self.ui.ComboBoxTarget.addItem(self.parent.ui.EEGFilesList.item(i).text())
        self.ui.ComboBoxTarget.currentIndexChanged.connect(self.scroll_target_set)
        #####################
        # target scroll
        self.rbs_target = []
        self.target_evt_onset = []
        self.target_evt_description = []
        self.ui.ComboBoxTarget.setCurrentIndex(-1)
        self.ui.ComboBoxTarget.setCurrentIndex(0)
        ####################
        # double spin box
        self.ui.DoubleSpinBoxWindowTime.setMinimum(0.1)
        self.ui.DoubleSpinBoxWindowTime.setMaximum(99.99)
        self.ui.DoubleSpinBoxWindowTime.setValue(0.5)
        ####################
        # channels setting
        self.ui.PushButtonChangeChannels.clicked.connect(self.change_seed_target)

        self.chan_name = self.raw_seed.ch_names
        self.chan_nums = len(self.chan_name)
        self.psi_seed = self.chan_name[: int(self.chan_nums / 2)]
        self.psi_target = self.chan_name[int(self.chan_nums / 2):]

        self.ui.LabelSeed.setText("")
        self.ui.LabelTarget.setText("")
        ####################
        # permutaion times
        self.ui.SpinBoxPermTimes.setMinimum(1000)
        self.ui.SpinBoxPermTimes.setMaximum(20000)
        self.ui.SpinBoxPermTimes.setValue(5000)
        ####################
        # shift time
        self.ui.DoubleSpinBoxShiftBefore.setMinimum(0)
        self.ui.DoubleSpinBoxShiftBefore.setMaximum(100)
        self.ui.DoubleSpinBoxShiftBefore.setValue(10)

        self.ui.DoubleSpinBoxShiftAfter.setMinimum(0)
        self.ui.DoubleSpinBoxShiftAfter.setMaximum(100)
        self.ui.DoubleSpinBoxShiftAfter.setValue(10)
        ####################
        # result
        self.ui.PushButtonResult.clicked.connect(self.calculate_result)
        self.ui.PushButtonResultPlot.clicked.connect(self.result_plot)
        self.ui.PushButtonResultSave.clicked.connect(self.result_save)
        self.ui.PushButtonResultPlot.setEnabled(False)
        self.ui.PushButtonResultSave.setEnabled(False)
        ####################

    def scroll_target_set(self, ind):
        self.rbs_target = []
        base_t = QWidget()
        base_t.setStyleSheet("background-color: lightyellow")
        layout_target = QVBoxLayout(base_t)

        self.raw_target = self.parent.raws[ind]
        self.target_evt_onset = self.raw_target.annotations.onset.tolist()
        self.target_evt_description = self.raw_target.annotations.description.tolist()

        for des in sorted(set(self.target_evt_description), key=self.target_evt_description.index):
            rb = QRadioButton(des)
            self.rbs_target.append(rb)
            layout_target.addWidget(rb)

        self.ui.ScrollAreaTarget.setWidget(base_t)

    def change_seed_target(self):
        sub = SubSeedTarget(self, radio=False)
        sub.show()

    def calculate_result(self):
        ######################################
        self.ui.LabelProgress.setText("parameter loading...")
        self.sfreq = self.raw_seed.info["sfreq"]

        event_seed = ""
        for rb in self.rbs_seed:
            if rb.isChecked():
                event_seed = rb.text()
                break

        event_seed_onset = [x for i, x in enumerate(self.seed_evt_onset) if self.seed_evt_description[i] == event_seed][
            0]

        event_target = ""
        for rb in self.rbs_target:
            if rb.isChecked():
                event_target = rb.text()
                break

        event_target_onset = \
        [x for i, x in enumerate(self.target_evt_onset) if self.target_evt_description[i] == event_target][0]

        self.window_t = self.ui.DoubleSpinBoxWindowTime.value()

        self.seed_channels = self.ui.LabelSeed.text().split(",")
        self.target_channels = self.ui.LabelTarget.text().split(",")
        self.seed_channels_ind = [i for i, x in enumerate(self.chan_name) if x in self.seed_channels]
        self.target_channels_ind = [i for i, x in enumerate(self.chan_name) if x in self.target_channels]

        permutation_times = self.ui.SpinBoxPermTimes.value()

        shift_minus = self.ui.DoubleSpinBoxShiftBefore.value() * -1
        shift_plus = self.ui.DoubleSpinBoxShiftBefore.value()

        # try:
        self.ui.LabelProgress.setText("MI calculation start")
        raw_arr_seed = self.raw_seed.get_data()
        raw_arr_targ = self.raw_target.get_data()
        self.p_values = np.empty((len(self.seed_channels), len(self.target_channels)))
        for k in self.seed_channels_ind:
            args = [(raw_arr_seed[[k, m], :], raw_arr_targ[m, :], event_seed_onset, event_target_onset, self.window_t,
                     shift_minus, shift_plus, permutation_times, self.sfreq) for m in self.target_channels_ind]
            with Pool(processes=os.cpu_count()) as p:
                ite = p.imap(permutation_mutual_information, args)
                for s, res in enumerate(ite):
                    self.ui.LabelProgress.setText("permutation test.. {}/{}".format(k + 1, len(self.seed_channels_ind)))
                    progress = s / len(self.target_channels_ind) * 100
                    self.ui.progressBar.setValue(progress)
                    app.processEvents()
                    self.p_values[k, s] = res

        self.ui.LabelProgress.setText("Finished")
        self.ui.progressBar.setValue(100)

        rej, self.p_fdr = mne.stats.fdr_correction(self.p_values)
        print(self.p_fdr)

        self.ui.PushButtonResultPlot.setEnabled(True)
        self.ui.PushButtonResultSave.setEnabled(True)

    def result_plot(self):
        fdr_bool = False
        if self.ui.ComboBoxFDR.currentIndex() == 0:
            fdr_bool = True

        if fdr_bool:
            fig, ax = matplotlib.pyplot.subplots(figsize=(self.p_fdr.shape[1] / 2, self.p_fdr.shape[0] / 2 + 1))
            sns.heatmap(self.p_fdr, annot=True, vmin=0, vmax=0.1, fmt='.2f', yticklabels=self.seed_channels,
                        xticklabels=self.target_channels, cmap="Reds_r", ax=ax)
            ax.set_ylim(self.p_fdr.shape[0], 0)
            ax.set_title("p_fdr MI")

            matplotlib.pyplot.show()

        else:
            fig, ax = matplotlib.pyplot.subplots(figsize=(self.p_values.shape[1] / 2, self.p_values.shape[0] / 2 + 1))
            sns.heatmap(self.p_values, annot=True, vmin=0, vmax=0.1, fmt='.2f', yticklabels=self.seed_channels,
                        xticklabels=self.target_channels, cmap="Reds_r", ax=ax)
            ax.set_ylim(self.p_values.shape[0], 0)
            ax.set_title("p_uncorrected MI")

            matplotlib.pyplot.show()

    def result_save(self):
        dir_path_save = ""
        try:
            dir_path = QFileDialog.getExistingDirectory(self, 'Select the Directory to save',
                                                        os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop")
            dir_path_save = dir_path + "/mi_result_{}".format(self.parent.ui.EEGFilesList.item(self.ind).text())
            os.makedirs(dir_path_save, exist_ok=False)
        except:
            self.ui.LabelProgress.setText("Processing was canceled because the directory with the same name could not be created")
            dir_path = ""

        if dir_path != "":
            np.save(dir_path_save + "/p_fdr", self.p_fdr)
            np.save(dir_path_save + "/p_uncorre", self.p_values)
            with open(dir_path_save + "/settings.txt", "w") as f:
                text = "settings: seed channels: {}, target channels: {}".format(self.seed_channels,
                                                                                 self.target_channels)
                f.write(text)

    def show(self):
        self.exec_()


class SubContinuousPlotting(QDialog):
    """
    Continuous PSI Plotting mode.
    """
    def __init__(self, parent, ind):
        super().__init__()
        self.ui = ui_continuous_plotting()
        self.ui.setupUi(self)
        self.setWindowTitle("Continuous PSI viewer")

        self.parent = parent
        self.ind = ind
        self.ui.PushButtonAllPlot.clicked.connect(self.all_plot)
        self.ui.buttonBox.accepted.connect(self.accepted)
        raw = self.parent.raws[self.ind]

        dir_path = self.parent.choiced_cont_dir

        self.cps_paths = glob.glob(dir_path + "/*-*.npy")
        ons_path = glob.glob(dir_path + "/onset.npy")
        settings_path = glob.glob(dir_path + "/settings.npy")

        self.st_names = []
        for i, cps_path in enumerate(self.cps_paths):
            cps_path = cps_path.strip(dir_path + "\\")
            cps_path = cps_path.strip(".npy")
            self.st_names.append(cps_path)
        # print(self.st_names)
        ############################
        # load
        self.ons = np.load(ons_path[0])
        settings = np.load(settings_path[0])
        self.sfreq = settings[0]
        self.window_time = settings[1]
        self.stride_t = settings[2]
        self.freqs = settings[3:]
        self.whole_t = math.ceil(raw.n_times / self.sfreq)
        # self.ons_s = self.ons / self.sfreq

        # print(self.freqs)
        # print(self.ons)

        ############################
        # slider and
        self.freq_range = [self.freqs[0], self.freqs[-1]]
        self.ui.HorizontalSliderFmin.setMinimum(self.freq_range[0] * 10)
        self.ui.HorizontalSliderFmin.setMaximum(self.freq_range[1] * 10)
        self.ui.HorizontalSliderFmax.setMinimum(self.freq_range[0] * 10)
        self.ui.HorizontalSliderFmax.setMaximum(self.freq_range[1] * 10)
        self.ui.HorizontalSliderFmin.setValue(self.freq_range[0] * 10)
        self.ui.HorizontalSliderFmax.setValue(self.freq_range[1] * 10)
        print(self.freq_range[0] * 10)
        print(self.freq_range[1] * 10)
        step = int(((self.freq_range[1] - self.freq_range[0]) / (len(self.freqs) - 1)) * 10)
        print(step)
        self.ui.HorizontalSliderFmin.setTickInterval(step)
        self.ui.HorizontalSliderFmax.setTickInterval(step)

        self.ui.HorizontalSliderFmin.setSingleStep(step)
        self.ui.HorizontalSliderFmax.setSingleStep(step)

        self.ui.HorizontalSliderFmin.setPageStep(step)
        self.ui.HorizontalSliderFmax.setPageStep(step)

        self.ui.LabelFmin.setText(str(self.freq_range[0]) + "Hz")
        self.ui.LabelFmax.setText(str(self.freq_range[1]) + "Hz")

        self.ui.HorizontalSliderFmin.valueChanged.connect(self.slider_changed)
        self.ui.HorizontalSliderFmax.valueChanged.connect(self.slider_changed)

        # combo box
        self.ui.ComboBoxSelectChannels.currentIndexChanged.connect(self.ch_changed)
        self.ui.ComboBoxSelectChannels.setCurrentIndex(0)

        def colcyc_auto():
            cl_stock = []
            while True:
                cc = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                if cc in cl_stock:
                    continue
                else:
                    cl_stock.append(cc)
                    yield cc


        self.color_auto = colcyc_auto()
        self.cl_dict = {}
        for i, st_name in enumerate(self.st_names):
            self.ui.ComboBoxSelectChannels.addItem(st_name)
            cc = self.color_auto.__next__()
            self.cl_dict[i] = cc
        # print(self.cl_dict)

        ############################
        # spin box
        self.ui.SpinBoxAverage.setValue(1)
        self.ui.SpinBoxAverage.setMinimum(1)
        self.ui.SpinBoxAverage.setMaximum(len(self.ons))

        # auto detect
        self.ui.PushButtonPeakDetect.clicked.connect(self.auto_peak_detect)

        ############################
        # window
        self.win = self.ui.graphicsView
        self.win.setBackground("w")
        self.win.setAntialiasing(aa=True)

        ############################
        # plot widget
        self.p0 = self.win.addPlot()
        self.p0.setMouseEnabled(y=False)
        self.p0.hideButtons()
        self.p0.showGrid(x=True, alpha=255)
        self.p0.setMenuEnabled(False)
        self.p0.setRange(yRange=(-0.1, 1.1), padding=0)
        self.p0.getAxis("bottom").setScale(1 / self.sfreq)
        self.p0.scene().sigMouseClicked.connect(self.mouse_clicked)

        self.plws = []
        self.all_plot = []
        self.all_plot_mode = False
        self.update_state()

        #############################
        self.win_sub = self.ui.graphicsView_sub
        self.win_sub.setBackground("w")

        self.p_sub = self.win_sub.addPlot()
        self.p_sub.setMouseEnabled(x=False)
        self.p_sub.setMouseEnabled(y=False)
        self.p_sub.plot(y=np.zeros(int(self.whole_t * self.sfreq)), pen=pqg.mkPen(color=(106, 13, 173), width=5))
        self.p_sub.setRange(xRange=(0, int(self.whole_t * self.sfreq)), yRange=(-1, 10))
        self.p_sub.getAxis("bottom").setScale(1 / self.sfreq)
        self.p_sub.hideAxis("left")
        self.p_sub.hideButtons()
        self.lrsub = pqg.LinearRegionItem((0, int(self.whole_t * self.sfreq)), brush=(255, 192, 203, 120))
        self.lrsub.sigRegionChanged.connect(self.update_plot)
        self.p0.sigXRangeChanged.connect(self.update_region)
        # self.p_sub.getAxis("bottom").setTickSpacing(major=100, minor=50)
        self.p_sub.addItem(self.lrsub)

        ##############################
        self.events_onset = raw.annotations.onset
        self.events_duration = raw.annotations.duration
        self.events_description = raw.annotations.description
        self.des_dict = {}
        for des in self.events_description:
            self.des_dict[des] = self.color_auto.__next__()

        for ons, des in zip(self.events_onset, self.events_description):
            ln = pqg.InfiniteLine(ons * self.sfreq, pen=[*self.des_dict[des], 120], label=des,
                                  labelOpts={"angle": 45, "color": self.des_dict[des]})
            ln_sub = pqg.InfiniteLine(ons * self.sfreq, pen=[*self.des_dict[des], 120])
            self.p0.addItem(ln)
            self.p_sub.addItem(ln_sub)

        self.evts_main = []
        self.evts_sub = []
        self.evts_ons = []
        self.evts_des = []
        ##############################
        # video_sync
        self.video_mode = False
        self.ui.PushButtonVideoSync.clicked.connect(self.video_sync)
        self.video_delay = 0
        self.video_playing = False
        ##############################

    def video_sync(self):
        if self.parent.ui.TextBrowserMovie.toPlainText() != "":

            self.video_delay = 0
            for x in self.events_description:
                if "vdist:" in x:
                    x = x.strip("vdist:")
                    if x.replace(".", "").isnumeric():
                        self.video_delay = float(x)
                    break

            self.sub = VideoWindow(self.parent.ui.TextBrowserMovie.toPlainText(), self, sync_mode=2)
            self.sub.show()
            self.video_playing = True
            # print(self.video_delay)
            self.ui.PushButtonVideoSync.setEnabled(False)

        else:
            pass

    def update_region(self):
        self.lrsub.setRegion(self.p0.getViewBox().viewRange()[0])

    def update_plot(self):
        self.p0.setXRange(*self.lrsub.getRegion(), padding=0)

    def update_state(self, mode=0):

        fmin_ind = self.freqs.tolist().index(self.ui.HorizontalSliderFmin.value() / 10)
        fmax_ind = self.freqs.tolist().index(self.ui.HorizontalSliderFmax.value() / 10)

        if self.all_plot_mode:
            try:
                for plw in self.all_plot:
                    self.p0.removeItem(plw)
            except:
                pass

            for i, cps_path in enumerate(self.cps_paths):
                data = np.load(cps_path)
                data_plot_all = np.mean(data[fmin_ind: fmax_ind + 1, :], axis=0)
                col = self.cl_dict[i]
                plw = self.p0.plot(x=self.ons, y=data_plot_all, pen=pqg.mkPen([*col, 40], width=2))
                self.all_plot.append(plw)

        try:
            for plw in self.plws:
                self.p0.removeItem(plw)
        except:
            pass

        self.plws = []
        sel_ind = self.ui.ComboBoxSelectChannels.currentIndex()

        if mode == 0:
            data = np.load(self.cps_paths[sel_ind])
            self.data_plot = np.mean(data[fmin_ind: fmax_ind + 1, :], axis=0)

        col = self.cl_dict[sel_ind]
        plw = self.p0.plot(x=self.ons, y=self.data_plot, pen=pqg.mkPen(col, width=3))
        self.plws.append(plw)

    def mouse_clicked(self, evt):
        pnt = QtCore.QPointF()
        pnt.setX(evt.pos()[0])
        pnt.setY(evt.pos()[1])
        pnt = self.p0.vb.mapToView(pnt)


        if evt.button() == 1:
            idx, pos = self.find_nearest(self.ons, pnt.x())
            print(pos)
            try:
                self.p0.removeItem(self.infln)
            except:
                pass
            self.infln = pqg.InfiniteLine(pos, pen=[75, 0, 130, 220], label="{:.2f}".format(float(self.data_plot[idx])),
                                          labelOpts={"angle": 45, "color": [75, 0, 130]})
            self.p0.addItem(self.infln)

            ###########################
            # video sync
            if self.video_playing:
                position = (pos / self.sfreq - self.video_delay) * 1000
                if position < 0:
                    position = 0
                self.sub.position_change(position)
            ###########################

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_A:
            mean_pnts = self.ui.SpinBoxAverage.value()
            self.mean_filter(mean_pnts)

            self.update_state(mode=1)
        if evt.key() == QtCore.Qt.Key_Up:
            ci = self.ui.ComboBoxSelectChannels.currentIndex()
            ci -= 1
            if ci < 0:
                ci = len(self.cps_paths) - 1
            self.ui.ComboBoxSelectChannels.setCurrentIndex(ci)
            self.update_state()

        if evt.key() == QtCore.Qt.Key_Down:
            ci = self.ui.ComboBoxSelectChannels.currentIndex()
            ci += 1
            if ci >= len(self.cps_paths):
                ci = 0
            self.ui.ComboBoxSelectChannels.setCurrentIndex(ci)
            self.update_state()

        if evt.key() == QtCore.Qt.Key_E:
            event_des = self.ui.TextEditEvent.toPlainText()
            if event_des != "":

                if event_des in self.des_dict.keys():
                    ev_col = self.des_dict[event_des]
                else:
                    ev_col = self.color_auto.__next__()
                    self.des_dict[event_des] = ev_col

                pos = self.infln.pos()
                print(pos / self.sfreq)

                ln = pqg.InfiniteLine(pos, pen=[*ev_col, 120], label=event_des,
                                      labelOpts={"angle": 45, "color": ev_col})
                ln_sub = pqg.InfiniteLine(pos, pen=[*ev_col, 120])
                self.p0.addItem(ln)
                self.p_sub.addItem(ln_sub)

                self.evts_main.append(ln)
                self.evts_sub.append(ln_sub)

                self.evts_ons.append(pos.x() / self.sfreq)
                self.evts_des.append(event_des)

        if evt.key() == QtCore.Qt.Key_D:
            pos = self.infln.pos()

            lnpos = [ln.pos() for ln in self.evts_main]
            if pos in lnpos:
                ind_del = lnpos.index(pos)
                self.p0.removeItem(self.evts_main[ind_del])
                self.p_sub.removeItem(self.evts_sub[ind_del])

                self.evts_main.pop(ind_del)
                self.evts_sub.pop(ind_del)

                self.evts_ons.pop(ind_del)
                self.evts_des.pop(ind_del)

            else:
                pass

    def accepted(self):
        if len(self.evts_ons) != 0:
            print(self.evts_ons)
            print(self.evts_des)
            self.parent.raws[self.ind].annotations.onset = np.append(self.parent.raws[self.ind].annotations.onset,
                                                                     np.array(self.evts_ons))
            self.parent.raws[self.ind].annotations.description = np.append(
                self.parent.raws[self.ind].annotations.description, np.array(self.evts_des))

            self.parent.raws[self.ind].annotations.duration = np.append(self.parent.raws[self.ind].annotations.duration,
                                                                        np.ones((len(self.evts_ons))))

    def gaussian_filter(self):
        gaussian1d = gaussian(2, std=2)
        self.data_plot = np.convolve(self.data_plot, gaussian1d, mode="same")
        print(self.data_plot)

    def mean_filter(self, n):
        self.data_plot = np.convolve(self.data_plot, np.ones(n) / float(n), "same")

    def ch_changed(self):
        self.update_state()

    def find_nearest(self, arr, v):
        idx = np.abs(arr - v).argmin()
        return idx, arr[idx]

    def slider_changed(self):
        if self.ui.HorizontalSliderFmin.value() > self.ui.HorizontalSliderFmax.value():
            self.ui.HorizontalSliderFmin.setValue(self.ui.HorizontalSliderFmax.value())
        if self.ui.HorizontalSliderFmax.value() < self.ui.HorizontalSliderFmin.value():
            self.ui.HorizontalSliderFmax.setValue(self.ui.HorizontalSliderFmin.value())

        self.ui.LabelFmin.setText(str(self.ui.HorizontalSliderFmin.value() / 10) + "Hz")
        self.ui.LabelFmax.setText(str(self.ui.HorizontalSliderFmax.value() / 10) + "Hz")

        self.update_state()

    def all_plot(self):
        if not self.all_plot_mode:
            self.all_plot_mode = True
            self.update_state()
        else:
            for plw in self.all_plot:
                self.p0.removeItem(plw)
            self.all_plot_mode = False
            self.update_state()

    def auto_peak_detect(self):
        self.sub_peak = SubPeakDetect(self)
        self.sub_peak.show()


class SubPeakDetect(QDialog):
    """
    Detect the peak in psi continuous plotting mode.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_auto_peak_detect()
        self.ui.setupUi(self)
        self.setWindowTitle("Auto Peak Detect")

        self.parent = parent

        # checkbox
        base = QWidget()
        base.setStyleSheet("background-color: honeydew")
        layout_seed = QGridLayout(base)
        self.cbs = []

        for i, st_name in enumerate(self.parent.st_names):
            cs = QCheckBox(st_name)
            self.cbs.append(cs)

            mm = i // 10
            nn = i % 10

            layout_seed.addWidget(cs, nn, mm)

        self.ui.scrollArea.setWidget(base)

        self.ui.SpinBoxOrder.setMinimum(1)
        self.ui.SpinBoxOrder.setMaximum(99)
        self.ui.SpinBoxOrder.setValue(1)

        self.ui.SpinBoxPreFilter.setMinimum(1)
        self.ui.SpinBoxPreFilter.setMaximum(999)
        self.ui.SpinBoxPreFilter.setValue(1)

        self.ui.PushButtonDetect.clicked.connect(self.detect_peak)
        self.ui.PushButtonClearResult.clicked.connect(self.clear_result)

        self.ui.comboBox.addItem("")  # 0
        self.ui.comboBox.addItem("Select All")  # 1
        self.ui.comboBox.addItem("Clear All")  # 2
        self.ui.comboBox.addItem("Frontal Group")  # 3 F
        self.ui.comboBox.addItem("Central Group")  # 4 C
        self.ui.comboBox.addItem("Parietal Group")  # 5 P
        self.ui.comboBox.addItem("Temporal Group")  # 6 T
        self.ui.comboBox.addItem("Occipital Group")  # 7 O

        self.ui.comboBox.setCurrentIndex(0)
        self.ui.comboBox.currentIndexChanged.connect(self.select_channels)

        self.ln_container = []
        self.lnsub_container = []

        self.ui.buttonBox.accepted.connect(self.accepted)
        self.ui.buttonBox.rejected.connect(self.rejected)

    def find_channels(self, word):
        for i, cb in enumerate(self.cbs):
            if word in cb.text():
                cb.setChecked(True)

    def select_channels(self):
        ind = self.ui.comboBox.currentIndex()

        word = ""
        if ind == 1:
            word = "All"
        elif ind == 2:
            word = "Clear"
        elif ind == 3:
            word = "F"
        elif ind == 4:
            word = "C"
        elif ind == 5:
            word = "P"
        elif ind == 6:
            word = "T"
        elif ind == 7:
            word = "O"

        if word == "All":
            for cb in self.cbs:
                cb.setChecked(True)
        elif word == "Clear":
            for cb in self.cbs:
                cb.setChecked(False)
        else:
            self.find_channels(word)

    def detect_peak(self):
        fmin_ind = self.parent.freqs.tolist().index(self.parent.ui.HorizontalSliderFmin.value() / 10)
        fmax_ind = self.parent.freqs.tolist().index(self.parent.ui.HorizontalSliderFmax.value() / 10)
        order = self.ui.SpinBoxOrder.value()
        prefilter = self.ui.SpinBoxPreFilter.value()

        sel_inds = []
        for i, cb in enumerate(self.cbs):
            if cb.isChecked():
                sel_inds.append(i)

        if sel_inds != []:
            data_arr = np.empty((0, self.parent.data_plot.shape[-1]))

            for sel_ind in sel_inds:
                data_tmp = np.load(self.parent.cps_paths[sel_ind])
                data_tmp = np.mean(data_tmp[fmin_ind: fmax_ind + 1, :], axis=0).reshape(1, -1)
                print(data_tmp.shape)
                data_arr = np.append(data_arr, data_tmp, axis=0)

            if prefilter > 1:
                for i in range(data_arr.shape[0]):
                    data_arr[i, :] = np.convolve(data_arr[i, :], np.ones(prefilter) / float(prefilter), "same")

            data_arr = np.mean(data_arr, axis=0).ravel()

            maxinds = argrelmax(data_arr, order=order)[0]
            self.ui.LabelComment.setText("{} peaks were found.".format(maxinds.shape[-1]))

            print(maxinds)
            if maxinds.shape[-1] > 0:
                for maxind in maxinds:

                    pos = self.parent.ons[maxind]
                    ln = pqg.InfiniteLine(pos, pen=pqg.mkPen([152, 251, 152, 160], width=3), label="found_peak",
                                          labelOpts={"angle": 45, "color": (152, 251, 152)})
                    ln_sub = pqg.InfiniteLine(pos, pen=[152, 251, 152, 160])

                    self.parent.p0.addItem(ln)
                    self.parent.p_sub.addItem(ln_sub)

                    self.ln_container.append(ln)
                    self.lnsub_container.append(ln_sub)

    def clear_result(self):
        for ln in self.ln_container:
            self.parent.p0.removeItem(ln)

        for lnsub in self.lnsub_container:
            self.parent.p_sub.removeItem(lnsub)

        self.ln_container = []
        self.lnsub_container = []

    def accepted(self):
        if self.ln_container != []:
            for ln in self.ln_container:
                self.parent.p0.removeItem(ln)
                ln_new = pqg.InfiniteLine(ln.pos(), pen=[165, 165, 165, 120], label="found_peak",
                                          labelOpts={"angle": 45, "color": [165, 165, 165]})
                self.parent.p0.addItem(ln_new)
                self.parent.evts_main.append(ln_new)
                self.parent.evts_ons.append(ln.pos() / self.parent.sfreq)
                self.parent.evts_des.append("found_peak")

            for lnsub in self.lnsub_container:
                self.parent.p_sub.removeItem(lnsub)
                lnsub_new = pqg.InfiniteLine(lnsub.pos(), pen=[165, 165, 165, 120])
                self.parent.p_sub.addItem(lnsub_new)
                self.parent.evts_sub.append(lnsub_new)
        self.close()

    def rejected(self):
        if self.ln_container != []:
            for ln in self.ln_container:
                print("aa")
                self.parent.p0.removeItem(ln)
            for lnsub in self.lnsub_container:
                self.parent.p_sub.removeItem(lnsub)
        self.close()


class SubPSILoad(QDialog):
    """
    continuous PSI load window.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_load_psi()
        self.ui.setupUi(self)
        self.setWindowTitle("Only load")

        self.parent = parent

        self.ui.PushButtonContinuousMode.clicked.connect(self.continuous_mode)
        # self.ui.PushButtonStatisticMode.clicked.connect(self.statistic_mode)

        self.parent.choiced = 0

        self.ui.PushButtonLoadFile.clicked.connect(self.load_dir)
        self.dir_path = ""

    def load_dir(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, 'Select the Directory that contains the target conutinous data',
                                                         os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop")
        self.ui.textBrowser.setText(self.dir_path)

    def continuous_mode(self):
        if self.dir_path != "":
            self.parent.choiced = 1
            self.parent.choiced_cont_dir = self.dir_path
            self.close()

    def show(self):
        self.exec_()


class SubPSIContinuous(QDialog):
    """
    Continuous PSI calculation window.
    """
    def __init__(self, parent, ind):
        super().__init__()
        self.ui = ui_continuous_psi()
        self.ui.setupUi(self)
        self.setWindowTitle("Continuous PSI calculation mode")

        self.parent = parent
        self.ind = ind

        self.raw = self.parent.raws[self.ind]

        self.ui.PushButtonResult.clicked.connect(self.calculation_psi)
        ###################
        # spin box

        self.ui.DoubleSpinBoxWindowTime.setMinimum(0.1)
        self.ui.DoubleSpinBoxWindowTime.setMaximum(9.9)
        self.ui.DoubleSpinBoxWindowTime.setDecimals(1)
        self.ui.DoubleSpinBoxWindowTime.setValue(4.0)

        self.ui.DoubleSpinBoxStrideTime.setMinimum(0.1)
        self.ui.DoubleSpinBoxStrideTime.setMaximum(9.9)
        self.ui.DoubleSpinBoxStrideTime.setDecimals(1)
        self.ui.DoubleSpinBoxStrideTime.setValue(1.0)

        self.ui.DoubleSpinBoxFreqMin.setMinimum(1.0)
        self.ui.DoubleSpinBoxFreqMin.setMaximum(99.9)
        self.ui.DoubleSpinBoxFreqMin.setDecimals(1)
        self.ui.DoubleSpinBoxFreqMin.setValue(5.0)

        self.ui.DoubleSpinBoxFreqMax.setMinimum(1.0)
        self.ui.DoubleSpinBoxFreqMax.setMaximum(99.9)
        self.ui.DoubleSpinBoxFreqMax.setDecimals(1)
        self.ui.DoubleSpinBoxFreqMax.setValue(31.0)

        self.ui.DoubleSpinBoxRangeStep.setMinimum(0.1)
        self.ui.DoubleSpinBoxRangeStep.setMaximum(9.9)
        self.ui.DoubleSpinBoxRangeStep.setDecimals(1)
        self.ui.DoubleSpinBoxRangeStep.setValue(1.0)

        #####################
        # channels setting
        self.ui.PushButtonChangeChannels.clicked.connect(self.change_seed_target)

        self.chan_name = self.raw.ch_names
        self.chan_nums = len(self.chan_name)
        self.psi_seed = self.chan_name[: int(self.chan_nums / 2)]
        self.psi_target = self.chan_name[int(self.chan_nums / 2):]

        self.ui.LabelSeed.setText(",".join(self.psi_seed))
        self.ui.LabelTarget.setText(",".join(self.psi_target))
        #####################

    def change_seed_target(self):
        sub = SubSeedTarget(self)
        sub.show()

    def calculation_psi(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select the directory to save. It is recommended to create a new Directory as it will be overwritten if the name is the same.',
                                                    os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop")

        if dir_path != "":
            self.ui.progressBar.setValue(0)
            self.ui.LabelProgress.setText("parameter loading...")
            ###############
            # parameters
            window_t = self.ui.DoubleSpinBoxWindowTime.value()
            stride_t = self.ui.DoubleSpinBoxStrideTime.value()
            seed_channels = self.ui.LabelSeed.text().split(",")
            seed_channels_ind = [i for i, x in enumerate(self.chan_name) if x in seed_channels]
            target_channels = self.ui.LabelTarget.text().split(",")
            target_channels_ind = [i for i, x in enumerate(self.chan_name) if x in target_channels]
            fmin = self.ui.DoubleSpinBoxFreqMin.value()
            fmax = self.ui.DoubleSpinBoxFreqMax.value()
            range_step = self.ui.DoubleSpinBoxRangeStep.value()
            ################。
            cwt_freqs = np.arange(fmin, fmax, range_step)
            cwt_n_cycles = cwt_freqs / 2
            sfreq = self.raw.info["sfreq"]

            raw_arr = self.raw.get_data()


            dir_path_save = dir_path + "/psi_{}".format(self.parent.ui.EEGFilesList.item(self.ind).text())
            os.makedirs(dir_path_save, exist_ok=True)

            # shared_arrays = []
            # for i in range(raw_arr.shape[0]):
            #     shared_arrays.append(Array("d", raw_arr[i, :].tolist()))
            # print(shared_arrays[0])

            ##########################
            # shared memory <- numpy
            n = raw_arr.astype(np.float32)

            #########
            # window onset time ind
            window_onset_time_ind = np.arange(0, n.shape[1] - window_t * sfreq, stride_t * sfreq)
            #########

            n_w, n_h = n.shape
            v = Value((ctypes.c_float * n_h) * n_w)
            v_c = v.get_obj()
            v_n = np.ctypeslib.as_array(v_c)
            v_n[:] = n
            ##########################

            args = [(v_n[seed, :], v_n[target, :], seed, target, window_t, stride_t, cwt_freqs, cwt_n_cycles, sfreq) for seed in
                    seed_channels_ind for target in target_channels_ind]
            print("start calculation")
            with Pool(processes=os.cpu_count()) as p:
                ite = p.imap(continuous_psi_async, args)
                for k, res in enumerate(tqdm(ite, total=len(args))):
                    self.ui.progressBar.setValue(100 * k / len(args))
                    self.ui.LabelProgress.setText("continuous PSI calculated...")
                    file_name = dir_path_save + "/{}-{}".format(self.chan_name[args[k][2]], self.chan_name[args[k][3]])
                    np.save(file_name, res)
            #########################

            np.save(dir_path_save + "/onset", window_onset_time_ind)

            setting_arr = np.hstack((np.array([sfreq, window_t, stride_t]), cwt_freqs))
            np.save(dir_path_save + "/settings", setting_arr)

            with open(dir_path_save + "/setting.txt", "w") as f:
                text = "settings.npy：Sfreq, window_time, stride_t, and the rest are cwt_freqs in that order."
                f.write(text)

            self.ui.LabelProgress.setText("Finished")
            self.ui.progressBar.setValue(100)
        else:
            self.ui.LabelProgress.setText("directory error")

    def show(self):
        self.exec_()


class SubPSIStatistic(QDialog):
    """
    Statistical PSI window.
    """
    def __init__(self, parent, ind):
        super().__init__()
        self.ui = ui_psi_statistical()
        self.ui.setupUi(self)
        self.setWindowTitle("Statistical PSI mode")

        self.parent = parent
        self.ind = ind

        self.raw_seed = self.parent.raws[self.ind]

        self.seed_evt_onset = self.raw_seed.annotations.onset.tolist()
        self.seed_evt_description = self.raw_seed.annotations.description.tolist()

        self.ui.LabelSeedCount.setText("0")
        self.ui.LabelTargetCount.setText("0")

        #####################
        # scroll seed
        base = QWidget()
        base.setStyleSheet("background-color: honeydew")
        layout_seed = QVBoxLayout(base)
        self.cbs_seed = []  # checkbox

        for des in sorted(set(self.seed_evt_description), key=self.seed_evt_description.index):
            cb = QCheckBox(des)
            cb.toggled.connect(self.seed_count)
            self.cbs_seed.append(cb)
            layout_seed.addWidget(cb)

        self.ui.ScrollAreaSeed.setWidget(base)
        #####################
        # target combo box
        for i in range(self.parent.ui.EEGFilesList.count()):
            self.ui.ComboBoxTarget.addItem(self.parent.ui.EEGFilesList.item(i).text())
        self.ui.ComboBoxTarget.currentIndexChanged.connect(self.scroll_target_set)
        #####################
        # target scroll
        self.cbs_target = []
        self.target_evt_onset = []
        self.target_evt_description = []
        self.ui.ComboBoxTarget.setCurrentIndex(-1)
        self.ui.ComboBoxTarget.setCurrentIndex(0)
        ####################
        # double spin box
        self.ui.DoubleSpinBoxWindowTime.setValue(1.00)
        self.ui.DoubleSpinBoxWindowTime.setMinimum(0.1)
        self.ui.DoubleSpinBoxWindowTime.setMaximum(99.99)
        ####################
        # channels setting
        self.ui.PushButtonChangeChannels.clicked.connect(self.change_seed_target)

        self.chan_name = self.raw_seed.ch_names
        self.chan_nums = len(self.chan_name)
        self.psi_seed = self.chan_name[: int(self.chan_nums / 2)]
        self.psi_target = self.chan_name[int(self.chan_nums / 2):]

        self.ui.LabelSeed.setText(",".join(self.psi_seed))
        self.ui.LabelTarget.setText(",".join(self.psi_target))
        ####################
        # Freq range
        self.ui.PushButtonAddRange.clicked.connect(self.add_freq_range)
        self.ui.PushButtonClearRange.clicked.connect(self.clear_freq_range)
        base_f = QWidget()
        base_f.setStyleSheet("background-color: aliceblue")
        self.layout_f = QVBoxLayout(base_f)
        self.layout_f.setAlignment(QtCore.Qt.AlignTop)
        self.labels_freq = []
        self.ui.ScrollAreaFreqRange.setWidget(base_f)
        ####################
        # fdr corrcetion
        ####################
        # permutaion times
        self.ui.SpinBoxPermTimes.setMinimum(1000)
        self.ui.SpinBoxPermTimes.setMaximum(20000)
        self.ui.SpinBoxPermTimes.setValue(5000)
        ####################
        # range step
        self.ui.DoubleSpinBoxRangeStep.setDecimals(1)
        self.ui.DoubleSpinBoxRangeStep.setValue(0.5)
        self.ui.DoubleSpinBoxRangeStep.setMinimum(0.1)
        self.ui.DoubleSpinBoxRangeStep.setMaximum(10.0)
        ####################
        # result
        self.ui.PushButtonResult.clicked.connect(self.calculation_result)
        self.ui.PushButtonResultPlot.clicked.connect(self.result_plot)
        self.ui.PushButtonResultSave.clicked.connect(self.result_save)
        self.ui.PushButtonResultPlot.setEnabled(False)
        self.ui.PushButtonResultSave.setEnabled(False)
        ####################
        # margin
        self.margin = [1.0, 1.0]
        ####################

    def scroll_target_set(self, ind):
        self.cbs_target = []
        base_t = QWidget()
        base_t.setStyleSheet("background-color: lightyellow")
        layout_target = QVBoxLayout(base_t)

        self.raw_target = self.parent.raws[ind]
        self.target_evt_onset = self.raw_target.annotations.onset.tolist()
        self.target_evt_description = self.raw_target.annotations.description.tolist()

        for des in sorted(set(self.target_evt_description), key=self.target_evt_description.index):
            cb = QCheckBox(des)
            cb.toggled.connect(self.target_count)
            self.cbs_target.append(cb)
            layout_target.addWidget(cb)

        self.ui.ScrollAreaTarget.setWidget(base_t)
        self.ui.LabelTargetCount.setText("0")
        self.ui.LabelComment.setText("Make the number of events the same for statistical tests.")

    def change_seed_target(self):
        sub = SubSeedTarget(self)
        sub.show()

    def seed_count(self):
        cnt = 0
        for i, cb in enumerate(self.cbs_seed):
            if cb.isChecked():
                tmp = len([x for x in self.seed_evt_description if x == cb.text()])
                cnt += tmp
        self.ui.LabelSeedCount.setText(str(cnt))

        target_int = int(self.ui.LabelTargetCount.text())
        if 0 < cnt == target_int > 0:
            self.ui.LabelComment.setText("ok")
        else:
            self.ui.LabelComment.setText("Make the number of events the same for statistical tests.")

    def target_count(self):
        cnt = 0
        for cb in self.cbs_target:
            if cb.isChecked():
                tmp = len([x for x in self.target_evt_description if x == cb.text()])
                cnt += tmp
        self.ui.LabelTargetCount.setText(str(cnt))

        seed_int = int(self.ui.LabelSeedCount.text())
        if 0 < cnt == seed_int > 0:
            self.ui.LabelComment.setText("ok")
        else:
            self.ui.LabelComment.setText("Make the number of events the same for statistical tests")

    def add_freq_range(self):
        freq_min, ok = QInputDialog.getDouble(self, "Frequency_min?", "Enter Frequency_min", 8.0, 0, 100, 1)
        if ok:
            freq_max, ok = QInputDialog.getDouble(self, "Frequency_max?", "Enter Frequency_max", freq_min, freq_min,
                                                  100, 1)
            if ok:
                label = QLabel("{} - {}".format(str(freq_min), str(freq_max)))
                label.setStyleSheet("background-color: lavenderblush")
                self.labels_freq.append(label)
                self.layout_f.addWidget(label)

    def clear_freq_range(self):
        def clearLayout(layout):
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    clearLayout(child.layout())

        self.labels_freq = []
        clearLayout(self.layout_f)

    def calculation_result(self):
        ##########################
        # parameters
        self.ui.LabelProgress.setText("parameter loading...")
        self.sfreq = self.raw_seed.info["sfreq"]

        self.events_seed = []
        for cb in self.cbs_seed:
            if cb.isChecked():
                self.events_seed.append(cb.text())
        events_seed_onset = [x for i, x in enumerate(self.seed_evt_onset) if
                             self.seed_evt_description[i] in self.events_seed]

        self.events_target = []
        for cb in self.cbs_target:
            if cb.isChecked():
                self.events_target.append(cb.text())
        events_target_onset = [x for i, x in enumerate(self.target_evt_onset) if
                               self.target_evt_description[i] in self.events_target]

        self.window_t = self.ui.DoubleSpinBoxWindowTime.value()  # window time

        self.seed_channels = self.ui.LabelSeed.text().split(",")  # seed channel
        self.target_channels = self.ui.LabelTarget.text().split(",")  # target channel
        self.seed_channels_ind = [i for i, x in enumerate(self.chan_name) if x in self.seed_channels]
        self.target_channels_ind = [i for i, x in enumerate(self.chan_name) if x in self.target_channels]

        self.fr_range = []
        for lb in self.labels_freq:
            self.fr_range.append(lb.text().split(" - "))  # frequency range [min, max] list

        permutation_times = self.ui.SpinBoxPermTimes.value()  # permutation
        self.step_fr = self.ui.DoubleSpinBoxRangeStep.value()
        ###########################
        if len(events_seed_onset) == len(events_target_onset):
            try:
                self.ui.LabelProgress.setText("psi calculation start")
                print("psi calculation start")
                task_psi = np.empty((len(events_seed_onset), len(self.fr_range), len(self.seed_channels_ind),
                                     len(self.target_channels_ind)))
                arg = [(x, self.raw_seed, self.fr_range, self.seed_channels_ind, self.target_channels_ind, self.margin,
                        self.window_t, self.step_fr, self.sfreq) for x in events_seed_onset]
                with Pool(processes=os.cpu_count()) as p:
                    ite = p.imap(psi_calc, arg)
                    for k, res in enumerate(tqdm(ite, total=len(arg))):
                        ################
                        # progress bar
                        self.ui.LabelProgress.setText("task_psi calculated...")
                        progress = k / len(events_seed_onset) * 100
                        self.ui.progressBar.setValue(progress)
                        app.processEvents()
                        ################
                        task_psi[k, :, :, :] = res
                self.ui.LabelProgress.setText("task_psi calculation finished")
                print("task_psi calculation finished")
                self.ui.progressBar.setValue(0)

                nontask_psi = np.empty((len(events_target_onset), len(self.fr_range), len(self.seed_channels_ind),
                                        len(self.target_channels_ind)))
                arg = [(x, self.raw_target, self.fr_range, self.seed_channels_ind, self.target_channels_ind, self.margin,
                       self.window_t, self.step_fr, self.sfreq) for x in events_target_onset]
                with Pool(processes=os.cpu_count()) as p:
                    ite = p.imap(psi_calc, arg)
                    for k, res in enumerate(tqdm(ite, total=len(arg))):
                        ################
                        # progress bar
                        self.ui.LabelProgress.setText("nontask_psi calculated...")
                        progress = k / len(events_seed_onset) * 100
                        self.ui.progressBar.setValue(progress)
                        app.processEvents()
                        ################
                        nontask_psi[k, :, :, :] = res

                self.ui.LabelProgress.setText("nontask_psi calculation finished")
                print("nontask_psi calculation finished")
                self.ui.progressBar.setValue(0)


                task_psi = np.transpose(task_psi, (1, 2, 3, 0))
                nontask_psi = np.transpose(nontask_psi, (1, 2, 3, 0))

                self.p_values = np.empty(
                    (len(self.fr_range), len(self.seed_channels_ind), len(self.target_channels_ind)))
                epoch_num = len(events_seed_onset)
                for l in range(len(self.fr_range)):
                    arg = [(self.target_channels_ind, task_psi[l, s, :, :], nontask_psi[l, s, :, :], permutation_times,
                            epoch_num) for s in range(len(self.seed_channels_ind))]
                    with Pool(processes=os.cpu_count()) as p:
                        ite = p.imap(permutation_test, arg)
                        for s, res in enumerate(tqdm(ite, total=len(arg))):
                            #################
                            # progress bar
                            self.ui.LabelProgress.setText("permutation test... {}/{}".format(l + 1, len(self.fr_range)))
                            progress = s / len(self.seed_channels_ind) * 100
                            self.ui.progressBar.setValue(progress)
                            app.processEvents()
                            #################
                            self.p_values[l, s, :] = res
                print(self.p_values.shape)

                rej, self.p_fdr = mne.stats.fdr_correction(self.p_values)
                print(self.p_fdr.shape)
                ####################
                # progress bar
                self.ui.LabelProgress.setText("Finished")
                print("Finished")
                self.ui.progressBar.setValue(0)
                self.ui.PushButtonResultPlot.setEnabled(True)
                self.ui.PushButtonResultSave.setEnabled(True)
                ####################
            except Exception as e:
                print(e.args)
                self.ui.LabelProgress.setText("Incorrect settings")

    def result_plot(self):
        fdr_bool = False
        if self.ui.ComboBoxFDR.currentIndex() == 0:
            fdr_bool = True


        if fdr_bool:
            plot_p(self.p_fdr, self.seed_channels, self.target_channels, self.fr_range, mode="plot", fdr=True)

        else:
            plot_p(self.p_values, self.seed_channels, self.target_channels, self.fr_range, mode="plot", fdr=False)

    def result_save(self):
        dir_path_save = ""
        try:
            dir_path = QFileDialog.getExistingDirectory(self, 'Select directory to save',
                                                        os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop")
            header, ok = QInputDialog.getText(self, 'event name?', 'Enter the event(save) name:', text="_event")
            dir_path_save = dir_path + "/st_result_{}".format(header)
            os.makedirs(dir_path_save, exist_ok=False)
        except:
            self.ui.LabelProgress.setText("Processing was canceled because the directory with the same name could not be created.")
            dir_path = ""


        if dir_path != "" and ok:
            np.save(dir_path_save + "/p_fdr", self.p_fdr)
            np.save(dir_path_save + "/p_uncorre", self.p_values)
            with open(dir_path_save + "/settings.txt", "w") as f:
                text = "settings: seed channels: {}, target channels: {}, seed_events: {}, target_events: {}, fr_range:{}".format(
                    self.seed_channels, self.target_channels, self.events_seed, self.events_target, self.fr_range)
                f.write(text)
            plot_p(self.p_fdr, self.seed_channels, self.target_channels, self.fr_range, mode="save", fdr=True, dirpath=dir_path_save + "/" + "fdr")
            plot_p(self.p_values, self.seed_channels, self.target_channels, self.fr_range, mode="save", fdr=False, dirpath=dir_path_save + "/" + "uncorre")


    def show(self):
        self.exec_()


class SubSyncSelf(QDialog):
    """
    Concatenate raw objects along time axis.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_sync_self()
        self.ui.setupUi(self)
        self.setWindowTitle("Concatenate data")

        self.parent = parent
        self.ui.buttonBox.accepted.connect(self.accepted)

        self.raw_length = []

        for raw in self.parent.raws:
            raw_sec = round(raw.n_times / raw.info["sfreq"], 2)
            self.raw_length.append(raw_sec)

        self.total_minimum = sum(self.raw_length) * 1.5

        self.q0s = []
        self.q1s = []

        t_sumtmp = 0
        width_sum = 0
        self.right_x = 20  # line right edge

        self.total_line_length = 1120
        for i, part_t in enumerate(self.raw_length):
            print(i)
            q0 = QLabel(parent=self)

            q1 = QDoubleSpinBox()

            #################
            width = self.total_line_length * part_t / self.total_minimum
            print(width)
            # q0.move(50, 50)
            q0.setGeometry(self.right_x + width_sum, 220, width, 30)

            # q0.setPageStep(int(part_t * 100))
            q0.setStyleSheet(
                "background-color: r({},{},{}); border-radius: 10; margin: 1".format(random.randint(0, 255),
                                                                                     random.randint(0, 255),
                                                                                     random.randint(0, 255)))

            #################
            # doublespinbox
            q1.setMinimum(0)
            q1.setMaximum(self.total_minimum * 10)
            q1.setDecimals(2)
            q1.setValue(t_sumtmp)

            width_sum += width
            t_sumtmp += part_t

            qv = QVBoxLayout()
            qv0 = QLabel()
            qv0.setText(self.parent.ui.EEGFilesList.item(i).text())
            qv.addWidget(qv0)
            qv.addWidget(q1)

            q1.valueChanged.connect(self.change_img)
            self.q1s.append(q1)
            self.q0s.append(q0)

            mm = i // 4
            nn = i % 4
            self.ui.gridLayout.addLayout(qv, mm, nn)

    def change_img(self):

        for i in range(len(self.q1s)):
            self.q0s[i].setGeometry(self.right_x + self.total_line_length * self.q1s[i].value() / self.total_minimum,
                                    self.q0s[i].y(),
                                    self.q0s[i].width(), self.q0s[i].height())

    def accepted(self):
        pnt = 0
        ch_names = self.parent.raws[0].ch_names
        sfreq = self.parent.raws[0].info["sfreq"]
        concat_arrs = []
        events_ons = []
        events_dur = []
        events_des = []
        err_flag = False
        for i in range(len(self.q1s)):
            v = self.q1s[i].value()
            anaume_length = v - pnt
            if v - pnt >= 0:
                print(v)
                print(pnt)

                anaume_arr = np.zeros((len(ch_names), int(anaume_length * self.parent.raws[i].info["sfreq"])))
                concat_arrs.append(anaume_arr)
                concat_arrs.append(self.parent.raws[i].get_data())

                events_ons += (self.parent.raws[i].annotations.onset + v).tolist()
                events_dur += self.parent.raws[i].annotations.duration.tolist()
                events_des += self.parent.raws[i].annotations.description.tolist()

                pnt = v + self.raw_length[i]

                # print(anaume_arr.shape)
                # print(self.parent.raws[i].get_data().shape)

            else:
                err_flag = True
                break

        if err_flag:
            self.parent.ui.TextBrowserComment.setText("Failed due to an error. Please try again.")

        else:
            new_arr = np.concatenate(concat_arrs, axis=1)
            # print(new_arr.shape)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
            new_raw = mne.io.RawArray(new_arr, info=info)

            # print(events_ons)
            # print(events_dur)
            # print(events_des)

            annot = mne.Annotations(events_ons, events_dur, events_des)
            new_raw.set_annotations(annot)

            self.parent.ui.EEGFilesList.addItem("raw_sync_with_movie")
            self.parent.raws.append(new_raw)

    def show(self):
        self.exec_()


class SubSyncMovie(QDialog):
    """
    Concatenate raw objects along time axis syncronized with video file.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_sync_movie()
        self.ui.setupUi(self)
        self.setWindowTitle("Concatenate data with movie")

        self.ui.gridLayout.setAlignment(QtCore.Qt.AlignTop)

        self.ui.buttonBox.accepted.connect(self.accepted)

        self.parent = parent

        self.raw_length = []

        for raw in self.parent.raws:
            raw_sec = round(raw.n_times / raw.info["sfreq"], 2)
            self.raw_length.append(raw_sec)

        self.total_minimum = max(sum(self.raw_length), self.parent.video_dur) * 1.5

        self.q0s = []
        self.q1s = []

        t_sumtmp = 0
        width_sum = 0
        self.right_x = 20

        self.total_line_length = 1120

        ##################
        # video plot
        self.qm = QLabel(parent=self)
        width = self.total_line_length * self.parent.video_dur / self.total_minimum
        self.qm.setGeometry(self.right_x, 330, width, 30)
        self.qm.setStyleSheet("background-color: r(136,72,152); border-radius: 10; margin: 1")

        self.qm1 = QDoubleSpinBox()
        self.qm1.setMinimum(0)
        self.qm1.setMaximum(self.total_minimum * 10)
        self.qm1.setDecimals(2)
        self.qm1.setValue(0)
        self.ui.horizontalLayout.addWidget(self.qm1)
        self.qm1.valueChanged.connect(self.change_img_movie)

        for i, part_t in enumerate(self.raw_length):
            print(i)
            q0 = QLabel(parent=self)

            q1 = QDoubleSpinBox()

            #################
            # texticon
            width = self.total_line_length * part_t / self.total_minimum
            # print(width)
            # q0.move(50, 50)
            q0.setGeometry(self.right_x + width_sum, 170, width, 30)

            q0.setStyleSheet(
                "background-color: r({},{},{}); border-radius: 10; margin: 1".format(random.randint(0, 255),
                                                                                     random.randint(0, 255),
                                                                                     random.randint(0, 255)))

            #################
            # doublespinbox
            q1.setMinimum(0)
            q1.setMaximum(self.total_minimum * 10)
            q1.setDecimals(2)
            q1.setValue(t_sumtmp)

            width_sum += width
            t_sumtmp += part_t

            qv = QVBoxLayout()
            qv0 = QLabel()
            qv0.setText(self.parent.ui.EEGFilesList.item(i).text())
            qv.addWidget(qv0)
            qv.addWidget(q1)

            q1.valueChanged.connect(self.change_img)
            self.q1s.append(q1)
            self.q0s.append(q0)

            mm = i // 4
            nn = i % 4
            self.ui.gridLayout.addLayout(qv, mm, nn)

    def slider_set(self, duration):
        self.duration_sec = duration / 1000
        print(self.duration_sec)

    def change_img(self):

        for i in range(len(self.q1s)):
            self.q0s[i].setGeometry(self.right_x + self.total_line_length * self.q1s[i].value() / self.total_minimum,
                                    self.q0s[i].y(),
                                    self.q0s[i].width(), self.q0s[i].height())

    def change_img_movie(self):
        self.qm.setGeometry(self.right_x + self.total_line_length * self.qm1.value() / self.total_minimum,
                            self.qm.y(), self.qm.width(), self.qm.height())

    def accepted(self):
        pnt = 0
        ch_names = self.parent.raws[0].ch_names
        sfreq = self.parent.raws[0].info["sfreq"]
        concat_arrs = []
        events_ons = []
        events_dur = []
        events_des = []
        err_flag = False
        for i in range(len(self.q1s)):
            v = self.q1s[i].value()
            anaume_length = v - pnt
            if v - pnt >= 0:
                # print(v)
                # print(pnt)

                anaume_arr = np.zeros((len(ch_names), int(anaume_length * self.parent.raws[i].info["sfreq"])))
                concat_arrs.append(anaume_arr)
                concat_arrs.append(self.parent.raws[i].get_data())

                events_ons += (self.parent.raws[i].annotations.onset + v).tolist()
                events_dur += self.parent.raws[i].annotations.duration.tolist()
                events_des += self.parent.raws[i].annotations.description.tolist()

                pnt = v + self.raw_length[i]

                # print(anaume_arr.shape)
                # print(self.parent.raws[i].get_data().shape)

            else:
                print(v -pnt)
                err_flag = True
                break

        if err_flag:
            self.parent.ui.TextBrowserComment.setText("Failed due to an error. Please try again.")

        else:
            ########################
            events_ons += [0]
            events_dur += [0.5]
            events_des += ["vdist:{}".format(self.qm1.value())]

            ########################

            new_arr = np.concatenate(concat_arrs, axis=1)
            # print(new_arr.shape)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
            new_raw = mne.io.RawArray(new_arr, info=info)

            print(events_ons)
            print(events_dur)
            print(events_des)

            annot = mne.Annotations(events_ons, events_dur, events_des)
            new_raw.set_annotations(annot)

            self.parent.ui.EEGFilesList.addItem("raw_sync_with_movie")
            self.parent.raws.append(new_raw)

    def show(self):
        self.exec_()


class LinearCustomHover(pqg.LinearRegionItem):
    """
    Override version of LinearRegionItem
    """
    def __init__(self, values=(0, 1), orientation='vertical', brush=None, pen=None,
                 hoverBrush=None, hoverPen=None, movable=True, bounds=None,
                 span=(0, 1), swapMode='sort', desc="", parent=""):
        super().__init__(values, orientation, brush, pen, hoverBrush, hoverPen, movable, bounds, span, swapMode)
        self.desc = desc
        self.parent = parent

    def hoverEvent(self, ev):
        self.parent.ui.label.setText(self.desc)


class SubViewerR(QDialog):
    """
    Viewer of EEG using pyqtgraph.
    """
    def __init__(self, parent, ind):
        super().__init__()
        self.ui = ui_eegview_focus()
        self.ui.setupUi(self)
        self.setWindowTitle("EEG Viewer")

        self.ui.PushButtonEvents.clicked.connect(self.event_checker)
        self.ui.PushButtonOption.clicked.connect(self.option)
        # self.ui.horizontalScrollBar.valueChanged.connect(self._move_h)
        # self.ui.horizontalScrollBar.setTracking(False)
        # self.ui.horizontalScrollBar.sliderReleased.connect(self._moved)
        self.ui.horizontalScrollBar.valueChanged.connect(self._move_both_h)
        self.ui.verticalScrollBar.valueChanged.connect(self._move_v)
        self.ui.PushButtonEventsEdit.clicked.connect(self.event_editor)
        self.ui.buttonBox.accepted.connect(self.accepted)
        self.ui.buttonBox.rejected.connect(self.rejected)
        self.ui.PushButtonPlayWithVideo.clicked.connect(self.play_with_video)
        self.ui.PushButtonPSISettings.clicked.connect(self.psi_settings)
        self.ui.PushButtonSpectogram.clicked.connect(self.show_spectrogram)
        self.ui.PushButtonInformationPlot.clicked.connect(self.information_plot)

        self.video_playing = False  # video flag
        self.parent = parent
        self.ind = ind
        raw = self.parent.raws[self.ind].copy()

        self.win = self.ui.graphicsView
        self.win.setBackground("w")
        self.win.setAntialiasing(aa=True)

        cc = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14E', '#EDC949', '#B07AA2', '#FF9DA7', '#9C755F',
              '#BAB0AC']


        def colcyc(color_cycle):
            i = 0
            le = len(color_cycle)
            while True:
                if i == le - 1:
                    i = 0
                yield color_cycle[i]
                i += 1

        def colcyc_auto():
            cl_stock = []
            while True:
                cc = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                if cc in cl_stock:
                    continue
                else:
                    cl_stock.append(cc)
                    yield cc

        ##################################
        raw.resample(60)
        self.raw_arr = raw.get_data()
        self.sfreq = raw.info["sfreq"]
        self.chan_name = raw.ch_names
        self.whole_t = math.ceil(self.raw_arr.shape[1] / self.sfreq)
        self.chan_n = len(self.chan_name)

        self.events_onset = raw.annotations.onset
        self.events_duration = raw.annotations.duration
        self.events_description = raw.annotations.description

        ###############################
        self.duration = 10
        self.channels_num = self.raw_arr.shape[0]
        self.channels_start = 0
        self.start_t = 0

        self.scale = np.percentile(self.raw_arr, 80) - np.percentile(self.raw_arr, 20)

        ###############################
        ##################################
        # channel check
        self.ch_sel = -1
        self.clcs = []
        self.bad_chs = raw.info["bads"]
        self.ch_rv = list(reversed(self.chan_name))
        self.ui.TextBrowserBad.setText(",".join(self.bad_chs))
        ##################################

        self.arr_plot = self.raw_arr / self.scale + np.arange(0, self.channels_num, 1).reshape(-1, 1)

        cc_cycle = colcyc(cc)

        self.p0 = self.win.addPlot()
        self.p0.setMouseEnabled(x=False)
        self.p0.setMouseEnabled(y=False)
        self.p0.setRange(xRange=(int(self.sfreq * self.start_t), int(self.sfreq * (self.start_t + self.duration))),
                         yRange=(self.channels_start - 1.0, self.channels_num), padding=0)
        self.p0.getAxis("bottom").setScale(1 / self.sfreq)
        self.p0.hideButtons()
        self.p0.showGrid(x=True, alpha=255)
        self.p0.setMenuEnabled(False)
        # y axis naming
        y_dict = {x: y for x, y in zip(reversed(list(range(self.channels_num))), self.chan_name)}
        self.p0.getAxis("left").setTicks([y_dict.items()])
        self.plws = []  # plot widget
        for i in range(self.channels_num):
            clc = cc_cycle.__next__()
            plw = self.p0.plot(y=self.arr_plot[i, :].ravel(), pen=pqg.mkPen(clc))
            self.plws.append(plw)
            self.clcs.append(clc)

        #################################
        self.ui.horizontalScrollBar.setRange(0, self.whole_t - self.duration)
        self.ui.verticalScrollBar.setRange(0, self.chan_n - self.channels_num)

        #####################################
        # sub view

        self.win_sub = self.ui.graphicsView_sub
        self.win_sub.setBackground("w")

        self.p_sub = self.win_sub.addPlot()
        self.p_sub.setMouseEnabled(x=False)
        self.p_sub.setMouseEnabled(y=False)
        self.p_sub.plot(y=np.zeros(int(self.whole_t * self.sfreq)), pen=pqg.mkPen(color=(106, 13, 173), width=5))
        self.p_sub.setRange(xRange=(0, int(self.whole_t * self.sfreq)), yRange=(-1, 10))
        # self.p_sub.hideAxis("bottom")
        self.p_sub.getAxis("bottom").setScale(1 / self.sfreq)
        self.p_sub.hideAxis("left")
        self.p_sub.hideButtons()
        self.lrsub = pqg.LinearRegionItem(
            (int(self.start_t * self.sfreq), int((self.start_t + self.duration) * self.sfreq)), movable=False,
            brush=(255, 192, 203, 120))
        # self.p_sub.getAxis("bottom").setTickSpacing(major=100, minor=50)
        self.p_sub.addItem(self.lrsub)
        #####################################

        ################################
        # event plot
        self.CA = colcyc_auto()

        self.dict_color = {x: self.CA.__next__() for x in set(self.events_description)}
        # dict_color_inv = dict((v, k) for k, v in self.dict_color.items())

        self.lrs = []
        self.lrs_sub = []
        self.labels = []

        for ons, dur, des in zip(self.events_onset, self.events_duration, self.events_description):

            cs = self.dict_color[des]
            range_tmp = (ons * self.sfreq, (ons + dur) * self.sfreq)
            lr = LinearCustomHover(range_tmp, brush=[*cs, 120], pen=[*cs, 200], desc=des, parent=self)
            lr.sigRegionChangeFinished.connect(self._update_range)
            lr_sub = pqg.LinearRegionItem(range_tmp, brush=[*cs, 120], pen=[*cs, 120])
            lr_sub.setMovable(False)
            self.lrs.append(lr)
            self.lrs_sub.append(lr_sub)
            self.p0.addItem(lr)
            self.p_sub.addItem(lr_sub)
            # text
            ti = pqg.TextItem(des, anchor=[0, 0.5], color=[*cs, 255])
            ti.setTextWidth(5)
            # ti.setParentItem(lr)
            # self.labels.append(ti)
            # self.p0.addItem(ti)

        ####################################
        # mouse event
        # proxy = pqg.SignalProxy(self.p0.scene().sigMouseMoved, rateLimit=60, slot=self.MouseMoved)
        self.p0.scene().sigMouseClicked.connect(self.mouse_clicked)
        ####################################
        self.selected_color = ""
        self.video_delay = 0

        ####################################
        # psi_settings
        self.psi_range = [8, 13]
        psi_seed_num = list(range(int(self.chan_n / 2)))
        self.psi_seed = [self.chan_name[i] for i in psi_seed_num]
        psi_target_num = list(range(int(self.chan_n / 2), int(self.chan_n)))
        self.psi_target = [self.chan_name[i] for i in psi_target_num]
        self.margin = [1.0, 1.0]
        self.psi_step = 0.5

        self.instant_psi_mode = False  # psi_mode flag
        ####################################

    def _update_range(self):
        for i, lr in enumerate(self.lrs):
            pos = lr.getRegion()
            # print(pos)
            self.lrs_sub[i].setRegion(pos)
            self.events_onset[i] = pos[0] / self.sfreq
            self.events_duration[i] = (pos[1] - pos[0]) / self.sfreq

        ###########
        # sub update
        try:
            self.sub_event._update()
        except:
            pass
        ###########

    def show_spectrogram(self):
        NFFT = 64
        matplotlib.pyplot.specgram(self.raw_arr[self.ch_sel, :], NFFT=NFFT, Fs=self.sfreq, noverlap=NFFT / 2,
                                   scale="linear")
        matplotlib.pyplot.title("Spectrogram of {}".format(self.ch_rv[self.ch_sel]))
        matplotlib.pyplot.ylabel("Hz")
        matplotlib.pyplot.xlabel("Time [s]")
        matplotlib.pyplot.show()

    def set_range_from_events(self):
        ont = self.events_onset * self.sfreq
        gen = (self.events_onset + self.events_duration) * self.sfreq
        for i in range(len(self.events_onset)):
            # print(ont[i])
            # print(gen[i])
            self.lrs[i].setRegion([ont[i], gen[i]])
            # self.lrs_sub[i].setRegion([ont[i], gen[i]])

    def play_with_video(self):
        """
        video sync
        """
        if self.parent.ui.TextBrowserMovie.toPlainText() != "":
            self.video_delay = 0
            for x in self.events_description:
                if "vdist:" in x:
                    x = x.strip("vdist:")
                    if x.replace(".", "").isnumeric():
                        self.video_delay = float(x)
                    break

            self.sub = VideoWindow(self.parent.ui.TextBrowserMovie.toPlainText(), self, sync_mode=1)
            self.sub.show()
            self.video_playing = True
            # print(self.video_delay)
            self.ui.PushButtonPlayWithVideo.setEnabled(False)

        else:
            pass

    def mouse_clicked(self, evt):
        pnt = QtCore.QPointF()
        pnt.setX(evt.pos()[0])
        pnt.setY(evt.pos()[1])
        pnt = self.p0.vb.mapToView(pnt)

        if evt.button() == 1:
            try:
                self.p0.removeItem(self.ln)
            except:
                pass
            self.ln = pqg.InfiniteLine(pnt, pen=[75, 0, 130, 220])
            self.p0.addItem(self.ln)

            #######################
            # video sync
            if self.video_playing:
                position = (pnt.x() / self.sfreq - self.video_delay) * 1000
                if position < 0:
                    position = 0
                self.sub.position_change(position)

            #######################
        if evt.button() == 2:
            if self.instant_psi_mode:
                try:
                    self.p0.removeItem(self.lr_psi)
                except:
                    pass
                self.lr_psi = LinearCustomHover((pnt.x(), pnt.x() + self.sfreq * 1.0), brush=[255, 255, 255, 120],
                                                pen=[255, 255, 255, 200], desc="psi", parent=self)
                self.p0.addItem(self.lr_psi)

            else:
                if self.selected_color != "":
                    # event add
                    # print(self.selected_color)
                    for k, v in self.dict_color.items():
                        if self.selected_color == v:
                            self.events_description = np.append(self.events_description, k)
                            range_tmp = (pnt.x(), pnt.x() + self.sfreq * 1.0)
                            lr = LinearCustomHover(range_tmp,
                                                   brush=[*self.selected_color, 120], pen=[*self.selected_color, 200],
                                                   desc=k, parent=self)
                            lr.sigRegionChangeFinished.connect(self._update_range)
                            self.p0.addItem(lr)
                            self.lrs.append(lr)

                            lr_sub = pqg.LinearRegionItem(range_tmp, brush=[*self.selected_color, 120],
                                                          pen=[*self.selected_color, 120])
                            lr_sub.setMovable(False)
                            self.p_sub.addItem(lr_sub)

                    self.events_onset = np.append(self.events_onset, pnt.x() / self.sfreq)
                    self.events_duration = np.append(self.events_duration, 1.0)

                    # event window update
                    try:
                        self.sub_event._update()
                    except:
                        pass

    def keyPressEvent(self, evt):

        if evt.key() == QtCore.Qt.Key_A:
            if not self.instant_psi_mode:
                self.instant_psi_mode = True
                self.ui.PushButtonOption.setEnabled(False)
                self.ui.PushButtonEvents.setEnabled(False)
                self.ui.PushButtonEventsEdit.setEnabled(False)
                self.ui.PushButtonPlayWithVideo.setEnabled(False)
                self.ui.PushButtonPSISettings.setEnabled(False)
                self.win.setBackground((200, 200, 200, 130))

            else:
                try:
                    self.p0.removeItem(self.lr_psi)
                except:
                    pass
                self.instant_psi_mode = False
                self.ui.PushButtonOption.setEnabled(True)
                self.ui.PushButtonEvents.setEnabled(True)
                self.ui.PushButtonEventsEdit.setEnabled(True)
                self.ui.PushButtonPlayWithVideo.setEnabled(True)
                self.ui.PushButtonPSISettings.setEnabled(True)
                self.win.setBackground("w")

        if evt.key() == QtCore.Qt.Key_C:

            try:
                range_min_sec, range_max_sec = self.lr_psi.getRegion()[0], self.lr_psi.getRegion()[1]
                range_min_sec = range_min_sec / self.sfreq
                range_max_sec = range_max_sec / self.sfreq

                if range_min_sec > self.margin[0] and range_max_sec < self.whole_t - self.margin[1]:
                    # margin
                    raw_calc = self.parent.raws[self.ind].copy()
                    sfreq_org = raw_calc.info["sfreq"]

                    psi_seed_num = [self.chan_name.index(x) for x in self.psi_seed]
                    psi_target_num = [self.chan_name.index(x) for x in self.psi_target]

                    cwt_freqs = np.arange(self.psi_range[0], self.psi_range[1], self.psi_step)

                    raw_calc.crop(range_min_sec - self.margin[0], range_max_sec + self.margin[1])

                    result_arr = np.empty((len(psi_seed_num), len(psi_target_num)))

                    for i, seed in enumerate(psi_seed_num):
                        for k, target in enumerate(psi_target_num):
                            psi_values = psi_morlet_by_arr(raw_calc.get_data(),
                                                           seed=seed, target=target, tmin_m=self.margin[0],
                                                           tmax_m=self.margin[1],
                                                           cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_freqs / 2,
                                                           sfreq=sfreq_org, faverage=True)
                            result_arr[i, k] = psi_values

                    # print(result_arr)
                    fig, ax = matplotlib.pyplot.subplots(figsize=(len(self.psi_target) / 2, len(self.psi_seed) / 2 + 1))
                    sns.heatmap(result_arr, vmin=0, vmax=1.0, annot=True, fmt='.2f', yticklabels=self.psi_seed,
                                xticklabels=self.psi_target,
                                cmap='Reds', ax=ax)
                    ax.set_ylim(len(self.psi_seed), 0)  # need
                    matplotlib.pyplot.show()
            except Exception as e:
                print(e.args)

        if evt.key() == QtCore.Qt.Key_Up:
            self.ch_sel += 1
            if self.ch_sel >= self.chan_n:
                self.plws[self.ch_sel - 1].setPen(self.clcs[self.ch_sel - 1])
                self.ch_sel = -1
            else:
                self.plws[self.ch_sel].setPen(self.clcs[self.ch_sel], width=3)
                self.plws[self.ch_sel - 1].setPen(self.clcs[self.ch_sel - 1])
        if evt.key() == QtCore.Qt.Key_Down:
            self.ch_sel -= 1
            if self.ch_sel < 0:
                self.plws[0].setPen(self.clcs[0])
                self.ch_sel = self.chan_n - 1
                self.plws[self.ch_sel].setPen(self.clcs[self.ch_sel], width=3)
            else:
                self.plws[self.ch_sel].setPen(self.clcs[self.ch_sel], width=3)
                self.plws[self.ch_sel + 1].setPen(self.clcs[self.ch_sel + 1])
        if evt.key() == QtCore.Qt.Key_B:

            if 0 <= self.ch_sel <= self.chan_n - 1:
                chsel = self.ch_rv[self.ch_sel]
                if chsel in self.bad_chs:
                    self.bad_chs.remove(chsel)
                else:
                    self.bad_chs.append(chsel)
                self.bad_chs = list(set(self.bad_chs))
                self.ui.TextBrowserBad.setText(",".join(self.bad_chs))

    def event_checker(self):
        self.sub_event = SubHyperEventEditor(self)
        self.sub_event.show()

    def event_editor(self):
        sub_e = SubSubEvent(self)
        sub_e.show()

    def _move_h(self, value):
        self.start_t = value
        range_view = int(self.sfreq * self.start_t), int(self.sfreq * (self.start_t + self.duration))
        # self.p0.setXRange(*range_view)
        self.lrsub.setRegion(range_view)
        self.hvalue = value

    def _moved(self):
        """
        This is faster than pre version
        """
        self.start_t = self.hvalue
        range_view = int(self.sfreq * self.start_t), int(self.sfreq * (self.start_t + self.duration))
        self.p0.setXRange(*range_view)

    def _move_both_h(self, value):
        self.start_t = value
        range_view = int(self.sfreq * self.start_t), int(self.sfreq * (self.start_t + self.duration))
        self.p0.setXRange(*range_view)
        self.lrsub.setRegion(range_view)
        self.hvalue = value

    def _move_v(self, value):
        self.channels_start = value
        self.p0.setYRange(self.chan_n - self.channels_start, self.chan_n - (self.channels_start + self.channels_num))

    def option(self):
        sub_o = SubSubOption(self)
        sub_o.show()

        # scroll bar update
        self.ui.horizontalScrollBar.setRange(0, self.whole_t - self.duration)
        self.ui.verticalScrollBar.setRange(0, self.chan_n - self.channels_num)
        self._move_h(self.ui.horizontalScrollBar.value())
        self._moved()
        # self._move_v(self.ui.verticalScrollBar.value())

    def psi_settings(self):
        sub = SubPSISettings(self)
        sub.show()

    def information_plot(self):
        self.sub_info = SubInformationPlot(self)
        self.sub_info.show()

    def accepted(self):

        self.parent.raws[self.ind].annotations.onset = self.events_onset
        self.parent.raws[self.ind].annotations.duration = self.events_duration
        self.parent.raws[self.ind].annotations.description = self.events_description

        bad_txt = self.ui.TextBrowserBad.toPlainText()
        if bad_txt != "":
            self.parent.raws[self.ind].info["bads"] = bad_txt.split(",")
        # print(self.parent.raws[self.ind].info["bads"])

        self.close()

    def rejected(self):
        self.close()


class SubInformationPlot(QDialog):
    """
    Instant information theory plotting mode.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_information_plot()
        self.ui.setupUi(self)
        self.parent = parent

        self.ui.PushButtonChangeEnt.clicked.connect(self.channels_select)
        self.checked_channels = []

        self.raw = self.parent.parent.raws[self.parent.ind]

        self.ui.PushButtonChangeSeed.clicked.connect(self.change_seed_target)
        self.ui.PushButtonChangeTarget.clicked.connect(self.change_seed_target)
        self.ui.PushButtonContinuousEntropyPlot.clicked.connect(self.cont_entropy_plot)
        self.ui.PushButtonContinuousEntropyPlotM.clicked.connect(self.cont_entropy_mplot)
        self.ui.PushButtonContinuousEntropyPlotML.clicked.connect(self.cont_entropy_mlplot)

        self.ui.DoubleSpinBoxWindowT.setMinimum(0.5)
        self.ui.DoubleSpinBoxWindowT.setMaximum(10.0)
        self.ui.DoubleSpinBoxWindowT.setDecimals(1)
        self.ui.DoubleSpinBoxWindowT.setValue(0.5)

        self.ui.DoubleSpinBoxStrideT.setMinimum(0.5)
        self.ui.DoubleSpinBoxStrideT.setMaximum(10.0)
        self.ui.DoubleSpinBoxStrideT.setDecimals(1)
        self.ui.DoubleSpinBoxStrideT.setValue(0.5)

        self.ui.DoubleSpinBoxWindowTM.setMinimum(0.5)
        self.ui.DoubleSpinBoxWindowTM.setMaximum(10.0)
        self.ui.DoubleSpinBoxWindowTM.setDecimals(1)
        self.ui.DoubleSpinBoxWindowTM.setValue(0.5)

        self.ui.DoubleSpinBoxStrideTM.setMinimum(0.5)
        self.ui.DoubleSpinBoxStrideTM.setMaximum(10.0)
        self.ui.DoubleSpinBoxStrideTM.setDecimals(1)
        self.ui.DoubleSpinBoxStrideTM.setValue(0.5)

        self.ui.DoubleSpinBoxWindowTML.setMinimum(0.5)
        self.ui.DoubleSpinBoxWindowTML.setMaximum(10.0)
        self.ui.DoubleSpinBoxWindowTML.setDecimals(1)
        self.ui.DoubleSpinBoxWindowTML.setValue(0.5)

        self.ui.SpinBoxL.setMinimum(1)
        self.ui.SpinBoxL.setMaximum(1000)
        self.ui.SpinBoxL.setValue(100)

        self.psi_seed = [self.parent.psi_seed[0]]
        self.psi_target = [self.parent.psi_target[0]]
        self.chan_name = self.parent.chan_name

        self.ui.LabelEnt.setText(str(self.psi_seed[0]))
        self.ui.LabelSeed.setText(self.psi_seed[0])
        self.ui.LabelTarget.setText(self.psi_target[0])

    def channels_select(self):

        self.checked_channels = []
        sub = SubDrop(self, self.chan_name)
        sub.show()
        self.ui.LabelEnt.setText(",".join(self.checked_channels))

    def change_seed_target(self):

        sub = SubSeedTarget(self, radio=True)
        sub.show()

    def cont_entropy_plot(self):
        tmin = 0
        tmax = self.parent.whole_t

        for ons, dur, des in zip(self.parent.events_onset, self.parent.events_duration, self.parent.events_description):
            if des == "m_info":
                tmin = ons
                tmax = ons + dur
                break

        chan_ind = [i for i, x in enumerate(self.chan_name) if x in self.ui.LabelEnt.text().split(",")]
        window_t = self.ui.DoubleSpinBoxWindowT.value()
        stride_t = self.ui.DoubleSpinBoxStrideT.value()

        entropy_continuous_calc(self.raw.get_data(), tmin, tmax, window_t, stride_t, chan_ind, self.chan_name,
                                plot_continuous=True, sfreq=self.raw.info["sfreq"])

    def cont_entropy_mplot(self):
        tmin = 0
        tmax = self.parent.whole_t

        for ons, dur, des in zip(self.parent.events_onset, self.parent.events_duration, self.parent.events_description):
            if des == "m_info":
                tmin = ons
                tmax = ons + dur
                break

        seed_ch_name = self.ui.LabelSeed.text()
        seed_ind = self.chan_name.index(seed_ch_name)

        target_ch_name = self.ui.LabelTarget.text()
        target_ind = self.chan_name.index(target_ch_name)

        window_t = self.ui.DoubleSpinBoxWindowTM.value()
        stride_t = self.ui.DoubleSpinBoxStrideTM.value()

        data_2d = self.raw.get_data()[[seed_ind, target_ind], :]

        mutual_information_continuous(data_2d, tmin=tmin, tmax=tmax, window_t=window_t, stride_t=stride_t,
                                      seed_ch_name=seed_ch_name,
                                      target_ch_name=target_ch_name, plot_continuous=True, sfreq=self.raw.info["sfreq"])

    def cont_entropy_mlplot(self):
        tmin = 0
        tmax = self.parent.whole_t

        for ons, dur, des in zip(self.parent.events_onset, self.parent.events_duration, self.parent.events_description):
            if des == "m_info":
                tmin = ons
                tmax = ons + dur
                break

        seed_ch_name = self.ui.LabelSeed.text()
        seed_ind = self.chan_name.index(seed_ch_name)

        target_ch_name = self.ui.LabelTarget.text()
        target_ind = self.chan_name.index(target_ch_name)

        lagged_window_t = self.ui.DoubleSpinBoxWindowTML.value()
        lagged_num = self.ui.SpinBoxL.value()

        data_2d = self.raw.get_data()[[seed_ind, target_ind], :]

        try:
            lagged_mutual_information(data_2d, tmin=tmin, tmax=tmax, lagtime_window_t=lagged_window_t,
                                      lag_nums=lagged_num, seed_ch_name=seed_ch_name, target_ch_name=target_ch_name,
                                      seed=0, plot=True, sfreq=self.raw.info["sfreq"])
        except Exception as e:
            print(e.args)


class SubPSISettings(QDialog):
    """
    Instant PSI calculation setting window.
    """
    def __init__(self, parent):

        super().__init__()
        self.ui = ui_psi_settings()
        self.ui.setupUi(self)

        self.parent = parent

        self.freq_min = self.parent.psi_range[0]
        self.freq_max = self.parent.psi_range[1]
        self.psi_step = self.parent.psi_step

        self.psi_seed = self.parent.psi_seed  # seed
        self.psi_target = self.parent.psi_target  # target
        self.chan_name = self.parent.chan_name

        self.ui.SpinBoxRangeMin.setValue(self.freq_min)
        self.ui.SpinBoxRangeMax.setValue(self.freq_max)
        self.ui.SpinBoxRangeMin.setMinimum(1)
        self.ui.SpinBoxRangeMax.setMinimum(1)

        self.ui.doubleSpinBox.setValue(self.psi_step)
        self.ui.doubleSpinBox.setMinimum(0.1)
        self.ui.doubleSpinBox.setMaximum(self.freq_max - self.freq_min)

        self.ui.LabelSeed.setText(",".join(self.psi_seed))
        self.ui.LabelTarget.setText(",".join(self.psi_target))

        self.ui.PushButtonChangeSeed.clicked.connect(self.change_seed_target)
        self.ui.PushButtonChangeTarget.clicked.connect(self.change_seed_target)

        self.ui.buttonBox.accepted.connect(self.accepted)

    def change_seed_target(self):
        sub = SubSeedTarget(self)
        sub.show()

    def accepted(self):
        if self.freq_min < self.freq_max:
            self.parent.psi_range[0] = self.freq_min
            self.parent.psi_range[1] = self.freq_max
            self.parent.psi_step = self.psi_step

        self.parent.psi_seed = self.psi_seed
        self.parent.psi_target = self.psi_target

    def show(self):
        self.exec_()


class SubSeedTarget(QDialog):
    """
    Seed and Target channels selecting window.
    """
    def __init__(self, parent, radio=False):
        super().__init__()
        self.ui = ui_seed_target()
        self.ui.setupUi(self)

        self.parent = parent

        self.checkboxes_seed = []
        self.checkboxes_target = []

        self.psi_seed = self.parent.psi_seed
        self.psi_target = self.parent.psi_target
        self.chan_name = self.parent.chan_name

        ######################
        # checkbox
        for i, ch in enumerate(self.chan_name):
            if not radio:
                cs = QCheckBox(str(ch))
                ct = QCheckBox(str(ch))
            else:
                cs = QRadioButton(str(ch))
                ct = QRadioButton(str(ch))

            if ch in self.psi_seed:
                cs.setChecked(True)
            if ch in self.psi_target:
                ct.setChecked(True)

            self.checkboxes_seed.append(cs)
            self.checkboxes_target.append(ct)

            mm = i // 32
            nn = i % 32

            self.ui.gridLayoutSeed.addWidget(cs, nn, mm)
            self.ui.gridLayoutTarget.addWidget(ct, nn, mm)

        ######################
        # buttons
        self.ui.PushButtonSelectAllSeed.clicked.connect(self.seed_select_all)
        self.ui.PushButtonClearSeed.clicked.connect(self.seed_clear)
        self.ui.PushButtonSelectAllTarget.clicked.connect(self.target_select_all)
        self.ui.PushButtonClearTarget.clicked.connect(self.target_clear)
        self.ui.buttonBox.accepted.connect(self.accepted)
        ######################

    def seed_select_all(self):
        for cs in self.checkboxes_seed:
            cs.setChecked(True)

    def seed_clear(self):
        for cs in self.checkboxes_seed:
            cs.setChecked(False)

    def target_select_all(self):
        for ct in self.checkboxes_target:
            ct.setChecked(True)

    def target_clear(self):
        for ct in self.checkboxes_target:
            ct.setChecked(False)

    def accepted(self):
        seed_num = []
        target_num = []
        for i in range(len(self.chan_name)):
            if self.checkboxes_seed[i].isChecked():
                seed_num.append(i)
            if self.checkboxes_target[i].isChecked():
                target_num.append(i)

        if seed_num == [] or target_num == []:
            pass
        else:
            self.parent.psi_seed = [self.chan_name[i] for i in seed_num]
            self.parent.psi_target = [self.chan_name[i] for i in target_num]
            self.parent.ui.LabelSeed.setText(",".join(self.parent.psi_seed))
            self.parent.ui.LabelTarget.setText(",".join(self.parent.psi_target))

    def show(self):
        self.exec_()


class SubHyperEventEditor(QDialog):
    """
    Event editor updated.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_hyper_event_editor()
        self.ui.setupUi(self)
        self.setWindowTitle("Event Editor")

        self.parent = parent

        self.ui.TextEdit.setText("psi_evt")
        self.ui.verticalLayoutLeft.setAlignment(QtCore.Qt.AlignTop)
        self.ui.verticalLayoutMiddle.setAlignment(QtCore.Qt.AlignTop)
        self.ui.verticalLayoutRight.setAlignment(QtCore.Qt.AlignTop)

        ##########################
        # buttons
        self.ui.PushButtonADDEvt.clicked.connect(self.add_event)
        self.ui.PushButtonDeleteEvt.clicked.connect(self.delete_event)
        self.ui.PushButtonUpdateState.clicked.connect(self.update_state)
        ##########################
        self.events_num = self.parent.events_onset.shape[-1]

        self.events_onset_wd = []
        self.events_duration_wd = []
        self.events_description_wd = []

        self._update()

    def _update(self):
        self.events_onset_wd = []
        self.events_duration_wd = []
        self.events_description_wd = []


        def clearLayout(layout):
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    clearLayout(child.layout())

        clearLayout(self.ui.verticalLayoutLeft)
        clearLayout(self.ui.verticalLayoutMiddle)
        clearLayout(self.ui.verticalLayoutRight)

        column_afford = 25

        for i, (ons, dur, des) in enumerate(
                zip(self.parent.events_onset, self.parent.events_duration, self.parent.events_description)):
            qh = QHBoxLayout()
            if len(des) > 30:
                des = des[-30:]
            qh0 = QRadioButton(des)
            if des in self.parent.dict_color.keys():
                col = self.parent.dict_color[des]
            else:
                col = self.parent.CA.__next__()
                self.parent.dict_color[des] = col

            tmp = max(col) + min(col)
            hosyoku = tuple([tmp - x for x in col])

            qh0.setStyleSheet("background-color: rgb{}; color: rgb{}".format(col, hosyoku))

            qh0.toggled.connect(self.choiced_color)
            print(ons)
            qh1 = QDoubleSpinBox()
            qh1.setDecimals(2)
            qh1.setMinimum(0)
            qh1.setMaximum(self.parent.whole_t)
            qh1.setValue(ons)
            qh2 = QDoubleSpinBox()
            qh2.setDecimals(2)
            qh2.setMinimum(0.01)
            qh2.setMaximum(self.parent.whole_t)
            qh2.setValue(dur)

            qh.addWidget(qh0, 5)
            qh.addWidget(qh1, 2)
            qh.addWidget(qh2, 2)

            #################
            # list adding
            self.events_description_wd.append(qh0)
            self.events_onset_wd.append(qh1)
            self.events_duration_wd.append(qh2)
            #################

            if i > column_afford * 2:
                self.ui.verticalLayoutRight.addLayout(qh)
            elif i > column_afford:
                self.ui.verticalLayoutMiddle.addLayout(qh)
            else:
                self.ui.verticalLayoutLeft.addLayout(qh)

        if len(self.events_description_wd) != 0:
            self.events_description_wd[-1].setChecked(True)

    def update_state(self):
        for i, (qh0, qh1, qh2) in enumerate(
                zip(self.events_description_wd, self.events_onset_wd, self.events_duration_wd)):
            self.parent.events_onset[i] = qh1.value()
            self.parent.events_duration[i] = qh2.value()
            self.parent.events_description[i] = qh0.text()

        # print(self.parent.events_onset)

        self.parent.set_range_from_events()

    def add_event(self):
        des = self.ui.TextEdit.toPlainText()
        self.parent.events_onset = np.append(self.parent.events_onset, 0)
        self.parent.events_duration = np.append(self.parent.events_duration, 0.5)
        self.parent.events_description = np.append(self.parent.events_description, des)

        self._update()

        range_tmp = (0, 0.5 * self.parent.sfreq)
        cs = self.parent.dict_color[des]
        lr = LinearCustomHover(range_tmp, brush=[*cs, 120], pen=[*cs, 200], desc=des, parent=self.parent)
        lr.sigRegionChangeFinished.connect(self.parent._update_range)
        self.parent.lrs.append(lr)
        self.parent.p0.addItem(lr)

        lr_sub = pqg.LinearRegionItem(range_tmp, brush=[*cs, 120], pen=[*cs, 120])
        lr_sub.setMovable(False)
        self.parent.lrs_sub.append(lr_sub)
        self.parent.p_sub.addItem(lr_sub)

        self.parent.set_range_from_events()

    def delete_event(self):
        checked_number = 0
        for i in range(len(self.events_description_wd)):
            if self.events_description_wd[i].isChecked():
                checked_number = i
                break

        self.parent.events_onset = np.delete(self.parent.events_onset, checked_number)
        self.parent.events_duration = np.delete(self.parent.events_duration, checked_number)
        self.parent.events_description = np.delete(self.parent.events_description, checked_number)
        self.parent.p0.removeItem(self.parent.lrs[checked_number])
        self.parent.lrs.pop(checked_number)

        self.parent.p_sub.removeItem(self.parent.lrs_sub[checked_number])
        self.parent.lrs_sub.pop(checked_number)

        self._update()
        self.parent.set_range_from_events()

    def choiced_color(self):
        checked_color = ""
        for i in range(len(self.events_description_wd)):
            if self.events_description_wd[i].isChecked():
                checked_color = self.parent.dict_color[self.parent.events_description[i]]
                break
        self.parent.selected_color = checked_color
        # print(self.parent.selected_color)


class SubSubEventChecker(QDialog):
    """
    Event checker.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_event_checker()
        self.ui.setupUi(self)
        self.ui.PushButtonRemove.clicked.connect(self.remove_event)

        self.parent = parent

        qf = QHBoxLayout()
        qf0 = QLabel()
        qf0.setText("color")
        qf1 = QLabel()
        qf1.setText("onset_time")
        qf2 = QLabel()
        qf2.setText("end_time")
        qf3 = QLabel()
        qf3.setText("comments")

        qf.addWidget(qf0, 2)
        qf.addWidget(qf1, 2)
        qf.addWidget(qf2, 2)
        qf.addWidget(qf3, 5)

        self.ui.verticalLayout.addLayout(qf)

        self.checkboxes = []

        for ons, dur, des in zip(self.parent.events_onset, self.parent.events_duration, self.parent.events_description):
            col = self.parent.dict_color[des]
            qh = QHBoxLayout()
            qh0 = QCheckBox()
            self.checkboxes.append(qh0)
            qh0.setStyleSheet("background-color: rgb{}".format(col))
            qh1 = QLabel()
            qh1.setText(str(round(ons, 2)) + "s")
            qh2 = QLabel()
            qh2.setText(str(round(ons + dur, 2)) + "s")
            qh3 = QLabel()
            if len(des) > 30:
                des = des[-30:]
            qh3.setText(des)

            qh.addWidget(qh0, 2)
            qh.addWidget(qh1, 2)
            qh.addWidget(qh2, 2)
            qh.addWidget(qh3, 5)

            self.ui.verticalLayout.addLayout(qh)

            self.dellist_ind = []
            self.ui.buttonBox.accepted.connect(self.accepted)

    def remove_event(self):
        self.dellist_ind = []
        for i, cb in enumerate(self.checkboxes):
            if cb.isChecked():
                self.dellist_ind.append(i)
        self.ui.label.setText(
            "{} will be removed".format([x for i, x in enumerate(self.parent.events_description) if i in self.dellist_ind]))

    def accepted(self):
        self.parent.dellist_ind = self.dellist_ind

    def show(self):
        self.exec_()


class SubSubEvent(QDialog):
    """
    Event checker
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_event()
        self.ui.setupUi(self)

        self.parent = parent

        self.colors = []
        self.radiobuttons = []

        qf = QHBoxLayout()
        qf0 = QLabel()
        qf0.setText("color")
        qf3 = QLabel()
        qf3.setText("comment")
        qf.addWidget(qf0, 2)
        qf.addWidget(qf3, 5)
        self.ui.verticalLayout.addLayout(qf)

        for ons, dur, des in zip(self.parent.events_onset, self.parent.events_duration, self.parent.events_description):
            col = self.parent.dict_color[des]
            if col in self.colors:
                pass
            else:
                self.colors.append(col)
                rb = QRadioButton()
                rb.setStyleSheet("background-color: rgb{}".format(col))
                self.radiobuttons.append(rb)
                qh = QHBoxLayout()
                qh.addWidget(rb, 2)
                q2 = QLabel()
                if len(des) > 50:
                    des = des[-50:]
                q2.setText(des)
                qh.addWidget(q2, 5)
                self.ui.verticalLayout.addLayout(qh)
                self.parent.selected_color = col

        self.ui.PushButtonAddNew.clicked.connect(self.add_new_event)

        tmp_ind = self.colors.index(self.parent.selected_color)
        self.radiobuttons[tmp_ind].setChecked(True)

        self.ui.buttonBox.accepted.connect(self.accepted)

    def add_new_event(self):
        name, ok = QInputDialog.getText(self, 'Event name', 'Enter new event name:')
        if name != "" and ok and not name in self.parent.events_description:
            new_cs = self.parent.CA.__next__()
            self.parent.dict_color[name] = new_cs

            rb = QRadioButton()
            rb.setStyleSheet("background-color: rgb{}".format(new_cs))
            qh = QHBoxLayout()
            qh.addWidget(rb, 2)
            q2 = QLabel()
            if len(name) > 50:
                name = name[-50:]
            q2.setText(name)
            qh.addWidget(q2, 5)

            self.ui.verticalLayout.addLayout(qh)
            self.colors.append(new_cs)
            self.radiobuttons.append(rb)

    def accepted(self):
        for i, rb in enumerate(self.radiobuttons):
            if rb.isChecked():
                print(i)
                self.parent.selected_color = self.colors[i]

    def show(self):
        self.exec_()


class SubSubOption(QDialog):
    """
    Viewer option window.
    """
    def __init__(self, parent):
        super().__init__()
        self.ui = ui_option()
        self.ui.setupUi(self)

        self.parent = parent

        self.ui.spinBox_c.setValue(parent.channels_num)
        self.ui.spinBox_s.setValue(parent.duration)

        self.ui.spinBox_c.setRange(1, parent.channels_num)
        self.ui.spinBox_s.setRange(1, parent.whole_t)

        self.ui.buttonBox.accepted.connect(self.accepted)

    def accepted(self):
        self.parent.channels_num = self.ui.spinBox_c.value()
        self.parent.duration = self.ui.spinBox_s.value()

    def show(self):
        self.exec_()


class VideoWindow(QDialog):
    """
    Video playing window.
    """
    def __init__(self, file_name, parent, sync_mode=False):
        super().__init__()
        self.ui = ui_video()
        self.ui.setupUi(self)
        self.setWindowTitle("Movie Viewer")
        self.parent = parent
        self.sync_mode = sync_mode
        # show front
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        # mediaplayer

        self.widget = QVideoWidget()
        self.ui.verticalLayout.addWidget(self.widget)

        self.mediaPlayer = QMediaPlayer(self)
        self.mediaPlayer.setVideoOutput(self.widget)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))

        # play icon
        self.ui.PushButtonPlay.setIcon(QtGui.QIcon("./images/play.png"))
        self.ui.PushButtonPlay.setIconSize(QtCore.QSize(35, 35))

        # volume icon
        self.ui.label.setPixmap(QPixmap("./images/volume.png").scaled(25, 25))
        # pause
        self.mediaPlayer.pause()

        self.ui.SliderVolume.setValue(50)
        self.duration = 0

        # signal
        self.mediaPlayer.setNotifyInterval(100)
        self.mediaPlayer.durationChanged.connect(self.slider_set)
        self.ui.PushButtonPlay.clicked.connect(self.control_video)
        self.ui.SliderVideo.valueChanged.connect(self.position_set)
        self.mediaPlayer.positionChanged.connect(self.position_change)
        self.ui.SliderVolume.sliderMoved.connect(self.volume_set)

    def control_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PausedState:
            self.mediaPlayer.play()
            self.ui.PushButtonPlay.setIcon(QtGui.QIcon("./images/pause.png"))
            self.ui.PushButtonPlay.setIconSize(QtCore.QSize(35, 35))

        else:
            self.mediaPlayer.pause()
            self.ui.PushButtonPlay.setIcon(QtGui.QIcon("./images/play.png"))
            self.ui.PushButtonPlay.setIconSize(QtCore.QSize(35, 35))

    def slider_set(self, duration):

        self.ui.SliderVideo.setRange(0, duration)
        self.duration = duration
        self.parent.video_dur = self.duration / 1000
        # print(self.parent.video_dur)

    def position_change(self, position):

        # print(position)
        self.ui.SliderVideo.setValue(position)

        seconds = position // 1000
        sec = self.duration // 1000
        self.ui.TimeDisp.setText("{}:{} / {}:{}".format(seconds // 60, seconds % 60, sec // 60, sec % 60))

        ########################
        # sync with eeg
        if self.sync_mode == 1:
            try:
                self.parent.p0.removeItem(self.parent.ln)
            except:
                pass
            # print("position:{}".format(position))
            pnt_x = (position / 1000 + self.parent.video_delay) * self.parent.sfreq
            # print("pnt_x:{}".format(pnt_x))
            self.parent.ln = pqg.InfiniteLine(pnt_x, pen=[255, 69, 0, 220])
            self.parent.p0.addItem(self.parent.ln)

            range_view = int(self.parent.sfreq * self.parent.start_t), int(
                self.parent.sfreq * (self.parent.start_t + self.parent.duration))
            if pnt_x < range_view[0] or range_view[1] < pnt_x:
                self.parent.start_t = math.floor(pnt_x / self.parent.sfreq)
                range_view = int(self.parent.sfreq * self.parent.start_t), int(
                    self.parent.sfreq * (self.parent.start_t + self.parent.duration))
                self.parent.p0.setXRange(*range_view)
                self.parent.lrsub.setRegion(range_view)

        ########################
        # continuous_plot sync
        if self.sync_mode == 2:
            try:
                self.parent.p0.removeItem(self.parent.ln)
            except:
                pass
            pnt_x = (position / 1000 + self.parent.video_delay) * self.parent.sfreq

            self.parent.ln = pqg.InfiniteLine(pnt_x, pen=[255, 69, 0, 220])
            self.parent.p0.addItem(self.parent.ln)

    def position_set(self, position):

        self.mediaPlayer.setPosition(position)

    def volume_set(self, position):
        self.mediaPlayer.setVolume(position)

    def closeEvent(self, evt):
        if self.sync_mode == 1:
            self.parent.video_playing = False
            self.parent.ui.PushButtonPlayWithVideo.setEnabled(True)
        if self.sync_mode == 2:
            try:
                self.parent.p0.removeItem(self.parent.ln)
            except:
                pass
            self.parent.video_playing = True
            self.parent.ui.PushButtonVideoSync.setEnabled(True)


class SubCheck(QDialog):
    """
    Check "OK" or not window.
    """
    def __init__(self, parent, text=""):
        super().__init__()
        self.parent = parent
        self.ui = ui_check()
        self.ui.setupUi(self)

        self.ui.label.setText(text)

        self.ui.buttonBox.accepted.connect(self.accepted)
        self.ui.buttonBox.rejected.connect(self.rejected)

    def accepted(self):
        self.parent.check = True
        print("ok")

    def rejected(self):
        self.parent.check = False
        print("cancel")

    def show(self):
        self.exec_()


class SubSplit(QDialog):
    """
    Split channels window.
    """
    def __init__(self, parent, text=""):
        super().__init__()
        self.parent = parent
        self.ui = ui_split()
        self.ui.setupUi(self)

        self.ui.TextBrowserBefore.setText(text)

        try:
            tmp = text.split(",")
            tmp = [x.strip("A_") for x in tmp]
            tmp = [x.strip("B_") for x in tmp]
            print(tmp)

            tmp = tmp[: int(len(tmp) / 2)]
            print(tmp)

            self.ui.TextBrowserA.setText(",".join(["A_" + x for x in tmp]))
            self.ui.TextBrowserB.setText(",".join(["B_" + x for x in tmp]))

        except:
            pass

        self.ui.CheckButton.clicked.connect(self.check_message)
        self.conv_cnt = 0
        self.ui.ConvButton.clicked.connect(self.conv_button)

        self.ui.buttonBox.accepted.connect(self.accepted)
        self.ui.buttonBox.rejected.connect(self.rejected)

    def check_message(self):
        chs_b = self.ui.TextBrowserBefore.toPlainText()
        chs_b = chs_b.split(",")

        chs_A = self.ui.TextBrowserA.toPlainText()
        chs_A = chs_A.split(",")

        chs_B = self.ui.TextBrowserB.toPlainText()
        chs_B = chs_B.split(",")

        if len(chs_b) != len(chs_A) + len(chs_B):
            self.ui.WarningLabel.setText("The length is different.")
            self.ui.buttonBox.setEnabled(False)
        else:
            self.ui.WarningLabel.setText("valid.")
            self.ui.buttonBox.setEnabled(True)

    def conv_button(self):
        if self.conv_cnt == 0:
            # template !!
            temp = ["Fp1", "Fz", "F3", "F7", "FC5", "FC1", "C3", "CP5", "CP1", "Pz", "P3",
                    "P7", "O1", "Oz", "O2", "P4", "P8", "CP6", "CP2", "Cz", "C4", "FC6",
                    "FC2", "F4", "F8", "Fp2", "ELL", "ERT", "ERU", "ERR"]
            A_chs = [str("A_") + x for x in temp]
            B_chs = [str("B_") + x for x in temp]

            self.ui.TextBrowserA.setText(",".join(A_chs))
            self.ui.TextBrowserB.setText(",".join(B_chs))
            self.conv_cnt += 1
        elif self.conv_cnt == 1:
            temp = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3',
                    'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                    'FC2', 'F4', 'F8', 'Fp2']
            self.ui.TextBrowserA.setText(",".join(temp))
            self.ui.TextBrowserB.setText(",".join(temp))
            self.conv_cnt -= 1

    def accepted(self):
        """
        ok
        """
        self.parent.split_channelsA = self.ui.TextBrowserA.toPlainText()
        self.parent.split_channelsB = self.ui.TextBrowserB.toPlainText()

    def rejected(self):
        """
        cancell
        """
        print("reject")

    def show(self):
        self.exec_()


class SubEOG(QDialog):
    """
    EOG channels selecting window
    """
    def __init__(self, parent, lis=[]):
        super().__init__()
        self.parent = parent
        self.ui = ui_eog()
        self.ui.setupUi(self)
        self.ui.buttonBox.accepted.connect(self.accepted)
        self.lis = lis

        self.checkboxes = []

        for i, ch in enumerate(lis):
            mm = i // 12
            nn = i % 12
            cb = QCheckBox(str(ch))
            self.checkboxes.append(cb)
            self.ui.gridLayout.addWidget(cb, nn, mm)

    def accepted(self):
        for i, cb in enumerate(self.checkboxes):
            if cb.isChecked():
                self.parent.eog_channels.append(self.lis[i])

        self.close()

    def show(self):
        self.exec_()


class SubDrop(QDialog):
    """
    Drop channels selecting window
    """
    def __init__(self, parent, lis=[]):
        super().__init__()
        self.parent = parent
        self.ui = ui_drop()
        self.ui.setupUi(self)
        self.ui.buttonBox.accepted.connect(self.accepted)
        self.lis = lis

        self.checkboxes = []

        for i, ch in enumerate(lis):
            mm = i // 12
            nn = i % 12
            cb = QCheckBox(str(ch))
            self.checkboxes.append(cb)
            self.ui.gridLayout.addWidget(cb, nn, mm)

    def accepted(self):
        for i, cb in enumerate(self.checkboxes):
            if cb.isChecked():
                self.parent.checked_channels.append(self.lis[i])

    def show(self):
        self.exec_()


class SubInfo(QDialog):
    """
    Detail Info of raw objects are shown in information window.
    """
    def __init__(self, text):
        super().__init__()
        self.ui = ui_sub()
        self.ui.setupUi(self)

        self.ui.textBrowser.setText(text)

    def show(self):
        self.exec_()


class SubRename(QDialog):
    """
    Rename channels window.
    """
    def __init__(self, parent, text=""):
        super().__init__()
        self.parent = parent
        self.ui = ui_rename()
        self.ui.setupUi(self)

        self.ui.TextBrowserBefore.setText(text)
        self.ui.TextBrowserAfter.setText(text)

        self.ui.CheckButton.clicked.connect(self.check_message)
        self.conv_cnt = 0
        self.ui.ConvButton.clicked.connect(self.conv_button)

        self.ui.buttonBox.accepted.connect(self.accepted)
        self.ui.buttonBox.rejected.connect(self.rejected)

    def check_message(self):
        chs_b = self.ui.TextBrowserBefore.toPlainText()
        chs_b = chs_b.split(",")

        chs = self.ui.TextBrowserAfter.toPlainText()
        chs = chs.split(",")
        if len(chs_b) != len(chs):
            self.ui.WarningLabel.setText("The length is different.")
            self.ui.buttonBox.setEnabled(False)
        else:
            self.ui.WarningLabel.setText("valid.")
            self.ui.buttonBox.setEnabled(True)

    def conv_button(self):
        if self.conv_cnt == 0:
            # template!!
            temp = ["Fp1", "Fz", "F3", "F7", "ELL", "FC5", "FC1", "C3", "ERU", "A1", "CP5", "CP1", "Pz", "P3",
                    "P7", "O1", "Oz", "O2", "P4", "P8", "A2", "CP6", "CP2", "Cz", "C4", "ERT", "ERR", "FC6",
                    "FC2", "F4", "F8", "Fp2"]
            A_chs = [str("A_") + x for x in temp]
            B_chs = [str("B_") + x for x in temp]
            ch_names = A_chs + B_chs
            self.ui.TextBrowserAfter.setText(",".join(ch_names))
            self.conv_cnt += 1
        elif self.conv_cnt == 1:
            temp = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3',
                    'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                    'FC2', 'F4', 'F8', 'Fp2']
            self.ui.TextBrowserAfter.setText(",".join(temp))
            self.conv_cnt -= 1

    def accepted(self):
        """
        ok
        """
        self.parent.ch_names_after = self.ui.TextBrowserAfter.toPlainText()

    def rejected(self):
        """
        cancelled
        """
        print("reject")

    def show(self):
        self.exec_()


class SubReorder(QDialog):
    """
    Reorder channels window.
    """
    def __init__(self, parent, text=""):
        super().__init__()
        self.parent = parent
        self.ui = ui_reorder()
        self.ui.setupUi(self)

        self.ui.TextBrowserBefore.setText(text)
        self.ui.TextBrowserAfter.setText(text)

        self.ui.CheckButton.clicked.connect(self.check_message)
        self.conv_cnt = 0
        self.ui.ConvButton.clicked.connect(self.conv_button)

        self.ui.buttonBox.accepted.connect(self.accepted)
        self.ui.buttonBox.rejected.connect(self.rejected)

    def check_message(self):
        chs_b = self.ui.TextBrowserBefore.toPlainText()
        chs_b = chs_b.split(",")

        chs = self.ui.TextBrowserAfter.toPlainText()
        chs = chs.split(",")
        if len(chs_b) != len(chs):
            self.ui.WarningLabel.setText("The length is different.")
            self.ui.buttonBox.setEnabled(False)
        else:
            self.ui.WarningLabel.setText("valid.")
            self.ui.buttonBox.setEnabled(True)

    def conv_button(self):
        if self.conv_cnt == 0:
            # テンプレ登録
            temp = ["Fp1", "Fz", "F3", "F7", "FC5", "FC1", "C3", "CP5", "CP1", "Pz", "P3",
                    "P7", "O1", "Oz", "O2", "P4", "P8", "CP6", "CP2", "Cz", "C4", "FC6",
                    "FC2", "F4", "F8", "Fp2", "A1", "A2", "ELL", "ERU", "ERT", "ERR"]
            A_chs = [str("A_") + x for x in temp]
            B_chs = [str("B_") + x for x in temp]
            ch_names = A_chs + B_chs
            self.ui.TextBrowserAfter.setText(",".join(ch_names))
            self.conv_cnt += 1
        elif self.conv_cnt == 1:
            temp = ["Fp1", "Fz", "F3", "F7", "FC5", "FC1", "C3", "CP5", "CP1", "Pz", "P3",
                    "P7", "O1", "Oz", "O2", "P4", "P8", "CP6", "CP2", "Cz", "C4", "FC6",
                    "FC2", "F4", "F8", "Fp2", "A1", "A2", "ELL", "ERU", "ERT", "ERR"]
            self.ui.TextBrowserAfter.setText(",".join(temp))
            self.conv_cnt -= 1

    def accepted(self):
        """
        ok
        """
        self.parent.ch_names_after = self.ui.TextBrowserAfter.toPlainText()

    def rejected(self):
        print("reject")

    def show(self):
        self.exec_()


class SubRef(QDialog):
    """
    Setting reference channels window.
    """
    def __init__(self, parent, text):
        super().__init__()
        self.parent = parent
        self.ui = ui_ref()
        self.ui.setupUi(self)
        self.text = text

        self.ui.PushButtonNoRef.clicked.connect(self.noref)
        self.ui.PushButtonAverage.clicked.connect(self.average)
        self.ui.PushButtonSElectrode.clicked.connect(self.electrodes)

    def noref(self):
        self.parent.chosen = "N"
        self.close()

    def average(self):
        self.parent.chosen = "A"
        self.close()

    def electrodes(self):
        self.parent.chosen = "M"
        self.chosens = ""
        sub = SubMultipleChoice(self, self.text)
        sub.show()

        self.parent.chosen_electrodes = self.chosens.split(",")
        self.close()

    def show(self):
        self.exec_()


class SubMultipleChoice(QDialog):
    """
    Select the shown choice window.
    """
    def __init__(self, parent, text):
        super().__init__()
        self.parent = parent
        self.ui = ui_choice()
        self.ui.setupUi(self)

        self.ui.TextEdit.setText(text)

        self.ui.buttonBox.accepted.connect(self.accepted)

    def accepted(self):
        self.parent.chosens = self.ui.TextEdit.toPlainText()

    def show(self):
        self.exec_()


###############################################################################################
###############################################################################################
# main
###############################################################################################
###############################################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyMainWindow()
    w.show()
    sys.exit(app.exec_())
