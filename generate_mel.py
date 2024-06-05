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
    
[y_piezo, fs] = load_filter_audio('data/test.flac')

# Create Dataset
df = pd.read_csv('data/test.txt', sep='\t', header=None)
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

    # Generate mask
    """stftMat = lr.stft(piezo_temp, 
                    n_fft= 1024, 
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann')"""
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

    abs_piezo_spec = np.abs(stftMat_mel)
    log_piezo = np.abs(10 * np.log10(abs_piezo_spec))

    # Apply binary thresholding
    threshold_value = 15
    _, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
    mask = 255 - mask
    mask = cv2.inRange(mask, 254, 255)
    mask = cv2.rotate(mask, cv2.ROTATE_180)
    mask = cv2.flip(mask, 1) # Flip horizontally
    
    
    normalized_image1, colormap1 = spec2dB(stftMat_mel) # Piezo
    
    cv2.imshow("mask_piezo", mask)
    cv2.imshow("test1",colormap1)
    cv2.imshow("test2",normalized_image1)
    #cv2.imwrite("temp_fig/mask.jpg", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#### ~0.27s for 255 frames. Try on your own