#Filtering only using the low pass filter 

from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff_freq, sampling_freq, order=4):
    nyquist=0.5*sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

filename = r'C:\Users\DIKSHA PANDEY\OneDrive\Desktop\IIITH_Internship\Research_Papers\ArtefactsMultiFrequencyBIA_Data Submission\TBME Data Submission\Bio-impedanceData.txt'
data = np.loadtxt(filename, delimiter=',')

time = data[:, 8]
bioimp = data[:, 9]

start_time = 20
end_time = 40

start_index = np.argmax(time >= start_time)
end_index = np.argmax(time >= end_time)

cutoff_frequency_low_breath = 0.4

sampling_frequency = 100

filtered_breath_data = butter_lowpass_filter(bioimp, cutoff_frequency_low_breath, sampling_frequency)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(time[start_index:end_index], bioimp[start_index:end_index], label='Original Data')
ax1.set_ylabel('bioimp')
ax1.legend()

ax2.plot(time[start_index:end_index], filtered_breath_data[start_index:end_index], label=f'Bandstop Filtered Breath Data ({cutoff_frequency_low_breath} Hz)', linestyle='-')
ax2.set_xlabel('time')
ax2.set_ylabel('bioimp')
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()