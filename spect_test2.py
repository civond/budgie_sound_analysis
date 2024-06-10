import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fftpack
import cv2
from utilities import *

# Load Audio
[y_piezo, fs] = load_filter_audio('data/bl122_piezo.flac')
[y_amb, fs] = load_filter_audio('data/bl122_amb.flac')

# Create Dataset
df = pd.read_csv('data/bl122_short.txt', sep='\t', header=None)
df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
df['onset_sample'] = (df['onset'] * fs).astype(int)
df['offset_sample'] = (df['offset'] * fs).astype(int)
df['length'] = df['offset_sample'] - df['onset_sample']

category_counts = df['label'].value_counts()
print(category_counts)
df = df[df['length'] >= 300]
df = df[df['label'] == 1]
df.reset_index(drop=True, inplace=True)

# Iterate across rows
for index, row in df.iterrows():
    start = int(row['onset_sample'])
    end = int(row['offset_sample'])

    piezo_temp = y_piezo[start:end]
    amb_temp = y_amb[start:end]

    # Generate spectrograms
    stftMat = gen_spec(piezo_temp, fs)
    stftMat2 = gen_spec(amb_temp, fs)

    # Create mask and apply binary thresholding
    abs_piezo_spec = np.abs(stftMat)**2
    log_piezo = np.abs(10 * np.log10(abs_piezo_spec))
    threshold_value = 30
    _, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
    mask = 255 - mask
    mask = cv2.inRange(mask, 254, 255)

    # Count number of white pixels
    num_pixels_0 = np.sum(mask == 0)
    num_pixels_255 = np.sum(mask == 255)
    print(f"Zeros: {num_pixels_0}, 255: {num_pixels_255}")
    height, width = mask.shape
    print(stftMat.shape)
    
    # Apply mask
    iStftMat, S_masked = apply_mask(stftMat, mask, fs)
    iStftMat2, S_masked2 = apply_mask(stftMat2, mask, fs)
    
    duration = len(piezo_temp)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)
    corr_val = np.correlate(iStftMat, iStftMat2)
    #print(corr_val)

    # Plotting
    plt.figure(1, figsize=(3,5))
    plt.subplot(3,1,1)
    plt.plot(t, piezo_temp, color='g', linewidth=1)
    plt.plot(t, iStftMat, color='b', linewidth=1)
    plt.plot(t, iStftMat2, color='r',linewidth=1)
    #plt.legend(['Orig','Piezo', 'Amb'], fontsize=8)
    plt.title(f"ISTFT Masked. Corr: {np.round(corr_val[0],4)}")
    plt.grid(True)
    plt.xlim(0,t[-1])
    plt.xlabel('Time (s)')


    corr_val = np.correlate(piezo_temp, iStftMat)
    
    plt.subplot(3,1,2)
    f1 = np.abs(fftpack.fft(iStftMat))
    f1 = np.fft.fftshift(f1)
    f2 = np.abs(fftpack.fft(iStftMat2))
    f2 = np.fft.fftshift(f2)
    freqs = fftpack.fftfreq(len(f1), 1/fs)
    freqs = np.fft.fftshift(freqs)
    
    plt.plot(freqs, f1, color='b')
    plt.plot(freqs, f2, color='r')
    plt.xlim(0,fs/2)
    plt.title("FFT")
    plt.ylabel('Mag.')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')

    #cv2.imshow("mask_piezo", mask)
    #cv2.imshow("piezo", np.abs(stftMat))
    #cv2.imshow("amb", np.abs(stftMat2))
    #cv2.imshow("piezo_recon", np.abs(S_masked))
    #cv2.imshow("piezo_recon2", np.abs(S_masked2))
    
    # Mask
    mask = cv2.rotate(mask, cv2.ROTATE_180)
    mask = cv2.flip(mask, 1) # Flip horizontally
    #cv2.imshow("mask_piezo", mask)
    cv2.imwrite("temp_fig/mask.jpg", mask)
    
    # Convert spectrograms
    normalized_image1, colormap1, temp1 = spec2dB(stftMat**2) # Piezo
    #cv2.imshow("piezo", colormap1)
    cv2.imwrite("temp_fig/orig_piezo.jpg", normalized_image1)
    cv2.imwrite("temp_fig/orig_piezo_cm.jpg", colormap1)

    normalized_image2, colormap2, temp2 = spec2dB(stftMat2**2) # Amb
    #cv2.imshow("amb", colormap2)
    cv2.imwrite("temp_fig/orig_amb.jpg", normalized_image2)
    cv2.imwrite("temp_fig/orig_amb_cm.jpg", colormap2)

    normalized_image3, colormap3, temp3 = spec2dB(S_masked**2) # Piezo_masked
    #cv2.imshow("masked_piezo", colormap3)
    cv2.imwrite("temp_fig/masked_piezo.jpg", normalized_image3)
    cv2.imwrite("temp_fig/masked_piezo_cm.jpg", colormap3)

    normalized_image4, colormap4, temp4 = spec2dB(S_masked2**2)   # Amb_masked
    #cv2.imshow("masked_amb", colormap4)
    cv2.imwrite("temp_fig/masked_amb.jpg", normalized_image4)
    cv2.imwrite("temp_fig/masked_amb_cm.jpg", colormap4)

    # Merge images for display
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    merged = np.concatenate((mask, colormap1, colormap2, colormap3, colormap4), axis=1)
    cv2.imshow(f"merged_{int(row['label'])}_{int(row['onset_sample'])}", merged)
    cv2.imwrite("temp_fig/merged_cm.jpg", merged)

    array = []
    height, width = normalized_image1.shape
    for i in range (width):
        p_column_values = temp3[:, i]
        a_column_values = temp4[:,i]
        rms = rmse(p_column_values, a_column_values)
        array.append(rms)
    
    plt.subplot(3,1,3)
    n = np.arange(0,len(array),1)
    rmse_mean = np.mean(array)
    window_size = 4
    ma = moving_average(array, window_size)
    #ma_n = np.arange(window_size - 1, len(array), 1)
    #plt.axhline(y=rmse_mean, color='r', linestyle='--')

    plt.plot(n, array,'g', alpha=0.3)
    plt.plot(n, ma, 'g', linestyle='--')
    plt.xlabel('STFT Bin')
    plt.ylabel('RMSE')
    plt.xlim(0,len(array)-1)
    plt.title(f"Spec. RMSE ({height}, {width})")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("temp_fig/figure.png")
    plt.clf()
    #plt.show()
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()