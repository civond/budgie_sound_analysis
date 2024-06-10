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
    db = lr.amplitude_to_db(np.abs(spec), ref=np.max)
    normalized_image = cv2.normalize(db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize
    normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_180) # Rotate 180 degrees
    normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    normalized_image = clahe.apply(normalized_image)
    

    """
    # Filtering
    F = np.fft.fft2(normalized_image)
    Fshift = np.fft.fftshift(F)

    [M,N] = normalized_image.shape
    order = 3

    #Cutoff Frequencies
    D0_low = 30
    D0_high = 15

    # Low Pass
    H1 = np.zeros((M,N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H1[u,v] = 1 / (1 + (D/D0_low)**order)
    #High Pass
    H2 = np.zeros((M,N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H2[u,v] = 1 / (1 + (D0_high/D)**order)
            
    # Combining filters
    H_bandpass = H1 * H2
    Gshift = Fshift * H_bandpass
    G = np.fft.ifftshift(Gshift)
    g_bandpass = np.abs(np.fft.ifft2(G) )

    result = normalized_image-g_bandpass
    result = np.clip(result, 0, 255).astype(np.uint8)
    print(result)

    cv2.imshow("test2",result)
    cv2.imshow("G", g_bandpass)
    cv2.imshow("BP", H_bandpass)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    if show_img == True:
        cv2.imshow("piezo", normalized_image)#, colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return normalized_image#, colormap

    
[y_piezo, fs] = load_filter_audio('data/bl122_piezo.flac')

# Create Dataset
df = pd.read_csv('data/bl122_short.txt', sep='\t', header=None)
df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
df['onset_sample'] = (df['onset'] * fs).astype(int)
df['offset_sample'] = (df['offset'] * fs).astype(int)
df['length'] = df['offset_sample'] - df['onset_sample']

category_counts = df['label'].value_counts()
print(category_counts)
df = df[df['length'] >= 300]
#df = df[df['label'] == 2]
df.reset_index(drop=True, inplace=True)
df.to_csv("meta.csv")
print(df)

# Iterate across rows
for index, row in df.iterrows():
    start = int(row['onset_sample'])
    end = int(row['offset_sample'])

    piezo_temp = y_piezo[start:end]
    temp = np.zeros(int(fs*0.255))
    """if len(piezo_temp) > int(fs*0.255):
         piezo_temp = piezo_temp[0:(int(fs*0.255))]
    else:
        temp[0:len(piezo_temp)] = piezo_temp
        piezo_temp = temp"""
    

    [mul, rem] = np.divmod(fs*0.254,len(piezo_temp))
    piezo_temp = np.tile(piezo_temp, int(mul+1))
    piezo_temp = piezo_temp[0:int(fs*0.224)]

    # Generate mask
    """stftMat_mel = lr.stft(piezo_temp, 
                    n_fft= 512, 
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann')"""
    stftMat_mel = lr.feature.melspectrogram(
                    y = piezo_temp, 
                    sr=fs,
                    n_fft=2048, 
                    win_length=int(fs*0.008),
                    hop_length=int(fs*0.001), 
                    center=True, 
                    window='hann',
                    n_mels=225,
                    fmax=10000)
    print(stftMat_mel.shape)

    abs_piezo_spec = np.abs(stftMat_mel)
    log_piezo = np.abs(10 * np.log10(abs_piezo_spec))
    
    
    normalized_image1 = spec2dB(stftMat_mel) # Piezo
    
    #cv2.imshow("mask_piezo", mask)
    #cv2.imshow("test1",colormap1)
    #cv2.imshow("test2",normalized_image1)
    cv2.imwrite(f"spectrograms/{int(row['onset_sample'])}.jpg", normalized_image1)
    #cv2.imwrite(f"mask/{int(row['onset_sample'])}.jpg", mask)
    #cv2.imwrite("temp_fig/mask.jpg", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#### ~0.27s for 255 frames. Try on your own
# https://colab.research.google.com/github/enzokro/clck10/blob/master/_notebooks/2020-09-10-Normalizing-spectrograms-for-deep-learning.ipynb#scrollTo=wCB9ye5aEXBE