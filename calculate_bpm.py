import numpy as np
import scipy as sci
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def bpm_calculation(green_channel):
    
    fps = len(green_channel)/15

    green_channel_mean = np.mean(green_channel)
    green_channel_std = np.std(green_channel)

    green_channel = green_channel - green_channel_mean
    green_channel = green_channel - green_channel_std

    low_cutoff = 0.0
    high_cutoff = 3.33

    b, a = butter(5, [low_cutoff, high_cutoff], btype='band')
    green_channel_filtered = lfilter(b, a, green_channel)

    green_channel_fft = np.abs(np.fft.fft(green_channel_filtered))
    green_channel_rfft = np.fft.rfftfreq(green_channel_filtered, d=1/fps)
    
    green_channel_fft = np.delete(green_channel_fft, 0)
    green_channel_fft_max = np.argmax(green_channel_fft)
    green_channel_hz = green_channel_rfft[np.where(green_channel_fft == green_channel_fft_max)]

    bpm = green_channel_hz * 60

    return bpm


    



