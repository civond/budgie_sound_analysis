import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fftpack
import cv2

# Filter audio
def load_filter_audio(path):
    [y, fs] = lr.load(path, sr=None)
    [b,a] = signal.cheby2(N=8,
                        rs=25,
                        Wn=300,
                        btype='high',
                        fs=fs)
    y = signal.filtfilt(b,a,y)
    return [y, fs]

# Create Spectrogram
def gen_spec(audio, fs):
     stftMat = lr.stft(audio, 
                    n_fft= 1024, 
                    win_length=int(fs*0.008),
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann')
     return stftMat

# Generate masked audio
def apply_mask(stftMat, mask, fs):
    real_part = stftMat.real * 32767
    imag_part = stftMat.imag * 32767 

    real_masked = cv2.bitwise_and(real_part, 
                                  real_part, 
                                  mask=mask)
    imag_masked = cv2.bitwise_and(imag_part, 
                                  imag_part, 
                                  mask=mask)
    # Convert the masked parts back to the original type
    real_masked = real_masked / 32767 
    imag_masked = imag_masked / 32767 

    S_masked = real_masked + 1j * imag_masked

    iStftMat = lr.istft(S_masked, 
                        n_fft=1024,
                        win_length= int(fs*0.008),
                        hop_length= int(fs*0.001), 
                        window='hann')
    return iStftMat, S_masked

# Convert spectrogramn to dB and rotate 180 degrees
def spec2dB(spec, show_img = False):
    ft_dB = lr.amplitude_to_db(np.abs(spec), ref=np.max)
    normalized_image = cv2.normalize(ft_dB, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize

    temp = ft_dB.copy()
    
    normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_180) # Rotate 180 degrees
    normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally
    colormap = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET) # Apply colormap
    if show_img == True:
            cv2.imshow("piezo", colormap)
    return normalized_image, colormap, temp

# Root Mean Square Error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Calculate Moving Average
def moving_average(data, window_size):
    vec = np.ones(window_size)/window_size
    ma = np.convolve(data, vec, 'same')
    return ma