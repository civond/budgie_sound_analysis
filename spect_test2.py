import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
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

# Generate masked audio
def apply_mask(stftMat, mask):
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
                        hop_length= int(fs*0.001), 
                        window='hann')
    return iStftMat, S_masked

# Convert spectrogramn to dB and rotate 180 degrees

def spec2dB(spec, show_img = False):
        ft_dB = lr.amplitude_to_db(np.abs(spec), ref=np.max)
        normalized_image = cv2.normalize(ft_dB, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize
        normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_180) # Rotate 180 degrees
        normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally
        colormap = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET) # Apply colormap
        if show_img == True:
             cv2.imshow("piezo", colormap)
        return normalized_image, colormap

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

    # Generate mask
    stftMat = lr.stft(piezo_temp, 
                    n_fft= 1024, 
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann')
    stftMat_mel = lr.feature.melspectrogram(
                    y = piezo_temp, 
                    sr=fs,
                    n_fft=2048, 
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann',
                    n_mels=255,
                    fmax=8000)
    print(stftMat_mel.shape)
    
    stftMat2 = lr.stft(amb_temp, 
                    n_fft= 1024, 
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann')

    abs_piezo_spec = np.abs(stftMat)
    log_piezo = np.abs(10 * np.log10(abs_piezo_spec))

    # Apply binary thresholding
    threshold_value = 15
    _, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
    mask = 255 - mask
    mask = cv2.inRange(mask, 254, 255)

    """#mask = cv2.rotate(mask, cv2.ROTATE_180)
    #mask = cv2.flip(mask, 1) # Flip horizontally
    stftMat = np.abs(stftMat)
    stftMat = cv2.rotate(stftMat, cv2.ROTATE_180)
    stftMat = cv2.flip(stftMat, 1) # Flip horizontally

    cv2.imshow("tesdt",np.abs(stftMat))
    cv2.imshow("test", stftMat_mel)
    cv2.imshow("mask_piezo", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    # Count number of white pixels
    num_pixels_0 = np.sum(mask == 0)
    num_pixels_255 = np.sum(mask == 255)
    print(f"Zeros: {num_pixels_0}, 255: {num_pixels_255}")
    height, width = mask.shape

    print(stftMat.shape)
    
    iStftMat, S_masked = apply_mask(stftMat, mask)
    iStftMat2, S_masked2 = apply_mask(stftMat2, mask)
    
    duration = len(piezo_temp)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)

    corr_val = np.correlate(iStftMat, iStftMat2)
    #print(corr_val)

    plt.figure(1, figsize=(6,4))
    plt.subplot(2,1,1)
    plt.title(f"Piezo-Amb Masked Corr: {np.round(corr_val[0],6)}")
    plt.plot(t, iStftMat, color='b')
    plt.plot(t, iStftMat2, color='r')
    plt.legend(['Piezo', 'Amb'])
    plt.grid(True)
    plt.xlim(0,t[-1])


    corr_val = np.correlate(piezo_temp, iStftMat)
    """import scipy.signal as signal
    corr_val = signal.correlate(piezo_temp, iStftMat, mode='full')
    norm = (corr_val-np.min(corr_val))/(np.max(corr_val)-np.min(corr_val))"""

    #normalized = (corr_val - np.min(corr_val)) / (np.max(corr_val) - np.min(corr_val))

    plt.subplot(2,1,2)
    plt.title(f"Piezo Corr: {np.round(corr_val[0],3)}")
    plt.plot(t, piezo_temp, color='g')
    plt.plot(t, iStftMat, color='b')
    plt.grid(True)
    plt.legend(['orig', 'masked'])
    plt.xlim(0,t[-1])
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("temp_fig/figure.png")
    plt.clf()

    #cv2.imshow("mask_piezo", mask)
    #cv2.imshow("piezo", np.abs(stftMat))
    #cv2.imshow("amb", np.abs(stftMat2))
    #cv2.imshow("piezo_recon", np.abs(S_masked))
    #cv2.imshow("piezo_recon2", np.abs(S_masked2))

    # Mask
    mask = cv2.rotate(mask, cv2.ROTATE_180)
    mask = cv2.flip(mask, 1) # Flip horizontally
    cv2.imshow("mask_piezo", mask)
    cv2.imwrite("temp_fig/mask.jpg", mask)
    
    # Convert spectrograms
    normalized_image1, colormap1 = spec2dB(stftMat) # Piezo
    #cv2.imshow("piezo", colormap1)
    cv2.imwrite("temp_fig/orig_piezo.jpg", normalized_image1)
    cv2.imwrite("temp_fig/orig_piezo_cm.jpg", colormap1)

    normalized_image2, colormap2 = spec2dB(stftMat2) # Amb
    #cv2.imshow("amb", colormap2)
    cv2.imwrite("temp_fig/orig_amb.jpg", normalized_image2)
    cv2.imwrite("temp_fig/orig_amb_cm.jpg", colormap2)

    normalized_image3, colormap3 = spec2dB(S_masked) # Piezo_masked
    #cv2.imshow("masked_piezo", colormap3)
    cv2.imwrite("temp_fig/masked_piezo.jpg", normalized_image3)
    cv2.imwrite("temp_fig/masked_piezo_cm.jpg", colormap3)

    normalized_image4, colormap4 = spec2dB(S_masked2)   # Amb_masked
    #cv2.imshow("masked_amb", colormap4)
    cv2.imwrite("temp_fig/masked_amb.jpg", normalized_image4)
    cv2.imwrite("temp_fig/masked_amb_cm.jpg", colormap4)

    # 2D correlation
    """corr2d = signal.correlate2d(normalized_image3, 
                                normalized_image4, 
                                boundary= 'symm', 
                                mode='same')
    cv2.imshow("corr", corr2d)"""
    """flatten_piezo = normalized_image3.flatten()
    flatten_amb = normalized_image4.flatten()
    corr2d = signal.correlate(flatten_piezo, flatten_amb, mode="same", method="fft")
    plt.subplot(2,1,1)
    plt.plot(flatten_piezo)
    plt.plot(flatten_amb)
    plt.subplot(2,1,2)
    plt.plot(corr2d)
    plt.clf()
    print(corr2d)
    print(corr2d.shape)"""

    # Merge images for display
    merged = np.concatenate((colormap1, colormap2, colormap3, colormap4), axis=1)
    cv2.imshow("merged", merged)
    cv2.imwrite("temp_fig/merged_cm.jpg", merged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()