import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fftpack
import cv2
from spec_utilities import *
import os

# Load Audio
"""[y_piezo, fs] = load_filter_audio('data/bl122_piezo.flac')
[y_amb, fs] = load_filter_audio('data/bl122_amb.flac')

# Create Dataset
df = pd.read_csv('data/bl122_short.txt', sep='\t', header=None)"""

# Load Audio
#[y_piezo, fs] = load_filter_audio('data/audioCh3_short_filtered.flac')
#[y_amb, fs] = load_filter_audio('data/audioCh1_short_filtered.flac')
[y_piezo, fs] = lr.load('data/audioCh3_short_filtered.flac', sr=None)
[y_amb, fs] = lr.load('data/audioCh1_short_filtered.flac', sr=None)

write_fig = False

# Create Dataset
df = pd.read_csv('data/audioCh3_short_filtered.txt', sep='\t', header=None)
df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
df['onset_sample'] = (df['onset'] * fs).astype(int)
df['offset_sample'] = (df['offset'] * fs).astype(int)
df['length'] = df['offset_sample'] - df['onset_sample']

category_counts = df['label'].value_counts()
print(category_counts)

df = df[df['label'] == 1]
df.reset_index(drop=True, inplace=True)

# Write directory
write_dir = "temp_fig"

# Iterate across rows
rows_to_drop_rmse = []
rows_to_drop_fft = []

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
    threshold_value = 45
    _, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
    mask = 255 - mask
    mask = cv2.inRange(mask, 254, 255)

    # Count number of white pixels
    num_pixels_0 = np.sum(mask == 0)
    num_pixels_255 = np.sum(mask == 255)
    #print(f"Zeros: {num_pixels_0}, 255: {num_pixels_255}")
    height, width = mask.shape
    print(f"\t{stftMat.shape}")
    
    # Apply mask
    iStftMat, S_masked = apply_mask(stftMat, mask, fs)
    iStftMat2, S_masked2 = apply_mask(stftMat2, mask, fs)
    
    duration = len(piezo_temp)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)
    #corr_val = np.correlate(iStftMat, iStftMat2)
    #print(corr_val)

    # Plotting
    #plt.subplot(2,1,1)
    #plt.plot(t, piezo_temp, color='g', linewidth=1)
    #plt.plot(t, iStftMat, color='b', linewidth=1)
    #plt.plot(t, iStftMat2, color='r',linewidth=1)
    #plt.legend(['Orig','Piezo', 'Amb'], fontsize=8)
    #plt.title(f"ISTFT Masked. Corr: {np.round(corr_val[0],4)}")
    #plt.grid(True)
    #plt.xlim(0,t[-1])
    #plt.xlabel('Time (s)')


    #corr_val = np.correlate(piezo_temp, iStftMat)

     # Mask
    mask = cv2.rotate(mask, cv2.ROTATE_180)
    mask = cv2.flip(mask, 1) # Flip horizontally

    # Convert spectrograms
    normalized_image1, colormap1, temp1 = spec2dB(stftMat**2) # Piezo
    normalized_image2, colormap2, temp2 = spec2dB(stftMat2**2) # Amb
    normalized_image3, colormap3, temp3 = spec2dB(S_masked**2) # Piezo_masked
    normalized_image4, colormap4, temp4 = spec2dB(S_masked2**2)  # Amb_masked

    array = []
    height, width = normalized_image1.shape
    for i in range (width):
        p_column_values = temp3[:, i]
        a_column_values = temp4[:,i]
        rms = rmse(p_column_values, a_column_values)
        array.append(rms)
    array_mean = np.mean(array)
    if array_mean > 30:
        rows_to_drop_rmse.append(index)

    f1 = np.abs(fftpack.fft(iStftMat))
    f1 = np.fft.fftshift(f1)
    f2 = np.abs(fftpack.fft(iStftMat2))
    f2 = np.fft.fftshift(f2)
    freqs = fftpack.fftfreq(len(f1), 1/fs)
    freqs = np.fft.fftshift(freqs)
    
    sum1 = np.sum(f1)/2
    sum2 = np.sum(f2)/2

    if sum2*8 < sum1:
        rows_to_drop_fft.append(index)

    print(sum1, sum2)

    if write_fig == True:
        temp_write = os.path.join(write_dir, str(int(row['onset_sample'])))
    
        try:
            os.mkdir(temp_write)
        except Exception as e:
            #print(f"{temp_write} already exists")
            pass

        plt.figure(1, figsize=(3,4))
        plt.subplot(2,1,1)

        plt.plot(freqs, f1, color='b')
        plt.plot(freqs, f2, color='r')
        plt.xlim(0,fs/2)
        plt.title("FFT")
        plt.ylabel('Mag.')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        
        # Mask
        cv2.imwrite(os.path.join(temp_write,"mask.jpg"), mask)

        # Im 1
        cv2.imwrite(os.path.join(temp_write,"orig_piezo.jpg"), normalized_image1)
        cv2.imwrite(os.path.join(temp_write,"orig_piezo_cm.jpg"), colormap1)
        
        # Im 2
        cv2.imwrite(os.path.join(temp_write,"orig_amb.jpg"), normalized_image2)
        cv2.imwrite(os.path.join(temp_write,"orig_amb_cm.jpg"), colormap2)

        # Im 3
        cv2.imwrite(os.path.join(temp_write,"masked_piezo.jpg"), normalized_image3)
        cv2.imwrite(os.path.join(temp_write,"masked_piezo_cm.jpg"), colormap3)

        # Im 4
        cv2.imwrite(os.path.join(temp_write,"masked_amb.jpg"), normalized_image4)
        cv2.imwrite(os.path.join(temp_write,"masked_amb_cm.jpg"), colormap4)

        # Merge images for display
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        merged = np.concatenate((mask, colormap1, colormap2, colormap3, colormap4), axis=1)
        #cv2.imshow(f"merged_{int(row['label'])}_{int(row['onset_sample'])}", merged)
        #cv2.imwrite("temp_fig/merged_cm.jpg", merged)
        cv2.imwrite(os.path.join(temp_write, "merged_cm.jpg"), merged)
        
        # RMSE plotting
        plt.subplot(2,1,2)
        n = np.arange(0,len(array),1)
        rmse_mean = np.mean(array)
        window_size = 4
        #ma = moving_average(array, window_size)
        #ma_n = np.arange(window_size - 1, len(array), 1)
        #plt.axhline(y=rmse_mean, color='r', linestyle='--')

        plt.plot(n, array,'g')
        plt.axhline(y=array_mean, linestyle="--", color='r')
        #plt.plot(n, ma, 'g', linestyle='--')
        plt.xlabel('STFT Bin')
        plt.ylabel('RMSE')
        plt.xlim(0,len(array)-1)
        plt.title(f"Spec. RMSE ({height}, {width})")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(temp_write,"figure.png"))
        plt.show()
        plt.clf()
            
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

# RMSE
print(f"Before drop: {len(df)}")
df_dropped_rmse = df.drop(index=rows_to_drop_rmse)
df_dropped_rmse.reset_index(drop=True, inplace=True)
print(f"After drop: {len(df_dropped_rmse)}")

columns_to_keep = ['onset', 'offset', 'label']
df_dropped_rmse = df_dropped_rmse.loc[:, columns_to_keep]
df_dropped_rmse.to_csv("dropped_rmse.txt", sep='\t', header=False, index=False)

# FFT
print(f"Before drop: {len(df)}")
df_dropped_fft = df.drop(index=rows_to_drop_fft)
df_dropped_fft.reset_index(drop=True, inplace=True)
print(f"After drop: {len(df_dropped_fft)}")

columns_to_keep = ['onset', 'offset', 'label']
df_dropped_fft = df_dropped_fft.loc[:, columns_to_keep]
df_dropped_fft.to_csv("dropped_fft.txt", sep='\t', header=False, index=False)