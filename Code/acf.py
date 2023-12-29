#code to generate the auto correlation function 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, find_peaks, correlate

def quality_label(rate):
    if rate >= 20:
        return 'poor'
    elif (15 <= rate < 20):
        return 'average'
    elif (10 <= rate < 15):
        return 'good'
    else:
        return 'excellent'

def compute_acf(signal):
    acf = correlate(signal, signal, mode='full')
    acf = acf / np.max(acf)
    lag = np.arange(-len(signal) + 1, len(signal))
    peaks, _ = find_peaks(acf)

    ap1 = ap2 = 0

    if len(peaks) > 0:
        ap1 = acf[peaks[0]]
        if len(peaks) > 1:
            ap2 = acf[peaks[1]]

    return lag, acf, ap1, ap2, peaks

def compute_segment_features(signal, fs, segment_size=60, sub_segment_size=15):
    acf_values = []
    num_segments = len(signal) // segment_size

    for i in range(num_segments):
        start_index = i * segment_size
        end_index = start_index + segment_size
        segment_signal = signal[start_index:end_index]
        _, acf_mean, _, _, _ = compute_acf_features(segment_signal, fs)
        acf_values.append(acf_mean)

    return acf_values

def compute_acf_features(signal, fs):
    lag, acf, ap1, ap2, peaks = compute_acf(signal)
    acf_mean = np.mean(acf)
    acf_std = np.std(acf)

    return acf_mean, acf_std

def butter_lowpass_filter(data, cutoff_freq, sampling_freq, order=4):
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff_freq, sampling_freq, order=8):
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    normal_cutoff = min(normal_cutoff, 0.99)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def filter_and_plot(time_vector, signal, title, low_cutoff, high_cutoff, fs, xlim=None, ylim=None):
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time_vector, signal, color='blue')
    plt.title(f'Original {title} Signal')
    plt.xlabel('time')
    plt.ylabel('bioimpedance')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.subplot(3, 1, 2)

    if 'Breath' in title:
        filtered_wave = butter_lowpass_filter(signal, low_cutoff, fs)
        label = 'Low-pass Filtered'
    elif 'Pulse' in title:
        filtered_wave = butter_highpass_filter(signal, high_cutoff, fs)
        label = 'High-pass Filtered'

    plt.plot(time_vector, filtered_wave, color='red', label=label)
    plt.title(f'{label} {title} Signal')
    plt.xlabel('time')
    plt.ylabel('bioimpedance')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.subplot(3, 1, 3)

    lag, acf, ap1, ap2, peaks = compute_acf(filtered_wave)

    if len(peaks) > 0:
        plt.plot(lag, acf, color='red', label='ACF Peaks')
        plt.plot(lag[peaks[0]], ap1, 'o', color='red', label='ACF Peak 1')
        if len(peaks) > 1:
            plt.plot(lag[peaks[1]], ap2, 'o', color='blue', label='ACF Peak 2')

    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(lag, acf, color='red', label='ACF Peaks')
    plt.title(f'AutoCorrelation Function ({title})')
    plt.xlabel('Lag')
    plt.ylabel('Normalized ACF')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.show()

fs = 100

file_path = r'C:\Users\DIKSHA PANDEY\OneDrive\Desktop\IIITH_Internship\Research_Papers\ArtefactsMultiFrequencyBIA_Data Submission\TBME Data Submission\Bio-impedanceData.txt'
data = np.genfromtxt(file_path, delimiter=',')

time_column = data[:, 8]
bioimpedance_column = data[:, 9]

time_vector = np.linspace(min(time_column), max(time_column), 200)
interp_bioimpedance = interp1d(time_column, bioimpedance_column, kind='linear', fill_value='extrapolate')
bioimpedance_at_time_vector = interp_bioimpedance(time_vector)
cosine_wave = np.cos(time_vector) * bioimpedance_at_time_vector

filter_and_plot(time_vector, cosine_wave, 'Breath', low_cutoff=0.4, high_cutoff=None, fs=fs, xlim=(0, 70), ylim=None)
filter_and_plot(time_vector, cosine_wave, 'Pulse', low_cutoff=None, high_cutoff=1, fs=fs, xlim=(20, 27), ylim=(-250, 250))