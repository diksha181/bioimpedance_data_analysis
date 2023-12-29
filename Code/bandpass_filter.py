#code to generate the breath and pulse rate corresponding to the given data

from scipy.signal import butter, sosfilt, find_peaks
import numpy as np
import matplotlib.pyplot as plt

def calculate_actual_rate(filtered_wave, peaks):
    time_diff = np.diff(peaks) / 100  
    breath_rate = 60 / np.mean(time_diff)
    return breath_rate

def butter_bandpass_filter(data, cutoff_freq_low, cutoff_freq_high, sampling_freq, order=6):
    nyquist = 0.5 * sampling_freq
    normal_cutoff_low = cutoff_freq_low / nyquist
    normal_cutoff_high = cutoff_freq_high / nyquist
    normal_cutoff_low = min(normal_cutoff_low, 0.99)
    normal_cutoff_high = min(normal_cutoff_high, 0.99)
    
    sos = butter(order, [normal_cutoff_low, normal_cutoff_high], btype='band', output='sos', analog=False)
    filtered_data = sosfilt(sos, data)
    return filtered_data

filename = r'C:\Users\DIKSHA PANDEY\OneDrive\Desktop\IIITH_Internship\Research_Papers\ArtefactsMultiFrequencyBIA_Data Submission\TBME Data Submission\Bio-impedanceData.txt'
data = np.loadtxt(filename, delimiter=',')

#referring to data calculated at 50kHz
time = data[:, 8]  
bioimp = data[:, 9]

start_time = 30
end_time = 60

start_index = np.argmax(time >= start_time)
end_index = np.argmax(time >= end_time)

cutoff_frequency_low_breath = 0.2
cutoff_frequency_high_breath = 0.3
cutoff_frequency_low_pulse = 1
cutoff_frequency_high_pulse = 2.4

sampling_frequency = 100

filtered_breath_data = butter_bandpass_filter(bioimp, cutoff_frequency_low_breath, cutoff_frequency_high_breath, sampling_frequency)
filtered_pulse_data = butter_bandpass_filter(bioimp, cutoff_frequency_low_pulse, cutoff_frequency_high_pulse, sampling_frequency)

peaks_breath, _ = find_peaks(filtered_breath_data)
peaks_pulse, _ = find_peaks(filtered_pulse_data)

breath_rate = calculate_actual_rate(filtered_breath_data, peaks_breath)
pulse_rate = calculate_actual_rate(filtered_pulse_data, peaks_pulse)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(time[start_index:end_index], bioimp[start_index:end_index], label='Original Data')
ax1.set_ylabel('bioimp')
ax1.legend()

ax2.plot(time[start_index:end_index], filtered_breath_data[start_index:end_index], label=f'Bandpass Filtered Breath Data ({cutoff_frequency_low_breath}-{cutoff_frequency_high_breath} Hz)', linestyle='--')
ax2.set_xlabel('time')
ax2.set_ylabel('bioimp')
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time[start_index:end_index], filtered_pulse_data[start_index:end_index], label=f'Bandpass Filtered Pulse Data ({cutoff_frequency_low_pulse}-{cutoff_frequency_high_pulse} Hz)', linestyle='-')
plt.title(f'Filtered data for {cutoff_frequency_low_pulse} Hz - {cutoff_frequency_high_pulse} Hz')
plt.xlabel('time')
plt.ylabel('bioimp')
#plt.legend()
plt.tight_layout()
plt.show()

print(f'Breath Rate: {breath_rate:.2f} bpm')
print(f'Pulse Rate: {pulse_rate:.2f} bpm')