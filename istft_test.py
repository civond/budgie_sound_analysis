import librosa as lr
import numpy as np
import matplotlib as plt

from spec_utilities import *

[y_piezo, fs] = load_filter_audio('data/piezo_short.flac')
[y_amb, fs] = load_filter_audio('data/amb_short.flac')


df = pd.read_csv('data/short_labels.txt', sep='\t', header=None)

# Load Audio
"""[y_piezo, fs] = load_filter_audio('data/bl122_piezo.flac')
[y_amb, fs] = load_filter_audio('data/bl122_amb.flac')

# Create Dataset
df = pd.read_csv('data/bl122_short.txt', sep='\t', header=None)"""
df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
df['onset_sample'] = (df['onset'] * fs).astype(int)
df['offset_sample'] = (df['offset'] * fs).astype(int)
df['length'] = df['offset_sample'] - df['onset_sample']

category_counts = df['label'].value_counts()
print(category_counts)
df = df[df['length'] >= 1000]
df = df[df['label'] == 1]
df.reset_index(drop=True, inplace=True)
print(df)

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
    print(f"Mask Shape: {mask.shape}")

    # Count number of white pixels
    num_pixels_0 = np.sum(mask == 0)
    num_pixels_255 = np.sum(mask == 255)
    print(f"Zeros: {num_pixels_0}, 255: {num_pixels_255}")
    height, width = mask.shape
    print(stftMat.shape)
    
    # Apply mask
    iStftMat, S_masked = apply_mask(stftMat, mask, fs)
    print(f"Masked shape: {S_masked.shape}")
    iStftMat2, S_masked2 = apply_mask(stftMat2, mask, fs)
    print(S_masked2.shape)

    duration = len(piezo_temp)/fs
    dt = 1/fs
    t = np.arange(0,duration,dt)
    corr_val = np.correlate(iStftMat, iStftMat2)
    #print(corr_val)

    mask = cv2.rotate(mask, cv2.ROTATE_180)
    mask = cv2.flip(mask, 1)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #cv2.imshow(f"merged_{int(row['label'])}_{int(row['onset_sample'])}", mask)
    normalized_image1, colormap1, temp1 = spec2dB(stftMat**2) # Piezo
    normalized_image2, colormap2, temp2 = spec2dB(stftMat2**2)
    normalized_image3, colormap3, temp3 = spec2dB(S_masked**2)
    normalized_image4, colormap4, temp4 = spec2dB(S_masked2**2)
    merged = np.concatenate((mask, colormap1, colormap2, colormap3, colormap4), axis=1)
    """cv2.imshow(f"merged_{int(row['label'])}_{int(row['onset_sample'])}", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    # Plotting
    plt.figure(1, figsize=(3,5))
    plt.subplot(3,1,1)
    plt.plot(t, piezo_temp, color='g', linewidth=1)
    plt.plot(t, iStftMat, color='b', linewidth=1)
    print(piezo_temp)
    print(iStftMat)
    #plt.plot(t, iStftMat2, color='r',linewidth=1)
    #plt.legend(['Orig','Piezo', 'Amb'], fontsize=8)
    plt.title(f"ISTFT Masked. Corr: {np.round(corr_val[0],4)}")
    plt.grid(True)
    plt.xlim(0,t[-1])
    plt.xlabel('Time (s)')
    plt.show()

#stftMat = gen_spec(y_piezo, fs)
"""istftMat = lr.istft(stftMat, 
                    n_fft=1024,
                    win_length= int(fs*0.008),
                    hop_length= int(fs*0.001), 
                    window='hann')"""


"""abs_piezo_spec = np.abs(stftMat)**2
log_piezo = np.abs(10 * np.log10(abs_piezo_spec))
threshold_value = 30
_, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
mask = 255 - mask
mask = cv2.inRange(mask, 254, 255)

iStftMat, S_masked = apply_mask(stftMat, mask, fs)

plt.figure(1)
plt.imshow(np.log10(np.abs(stftMat**2)))


plt.figure(2)
plt.plot(y_piezo)
plt.plot(iStftMat)
#plt.plot(y_piezo)
plt.show()"""