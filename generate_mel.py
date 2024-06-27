import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa as lr
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2
import os

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

# Convert spectrogramn to dB and rotate 180 degrees
def spec2dB(spec, spec2, mask, show_img = False):

    # Piezo Spectrogram
    db = lr.amplitude_to_db(np.abs(spec), ref=np.max)
    #db = db**2 # originally no squaring
    normalized_image = cv2.normalize(db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize
    normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_180) # Rotate 180 degrees
    normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    normalized_image = clahe.apply(normalized_image)

    # Amb Spectrogram
    db2 = lr.amplitude_to_db(np.abs(spec2), ref=np.max)
    db2 = db2 # originally no squaring
    normalized_image2 = cv2.normalize(db2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize
    normalized_image2 = clahe.apply(normalized_image2)

    normalized_image2 = cv2.bitwise_and(normalized_image2, 
                                  normalized_image2, 
                                  mask=mask)
    normalized_image2 = cv2.rotate(normalized_image2, cv2.ROTATE_180) # Rotate 180 degrees
    normalized_image2 = cv2.flip(normalized_image2, 1) # Flip horizontally

    # CLAHE

    #norm2color = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

    # I inputted this to test inputting into color channels
    rgb_image = np.zeros((normalized_image.shape[0], 
                          normalized_image.shape[1], 
                          3), dtype=np.uint8)
    
    rgb_image[:, :, 0] = normalized_image
    rgb_image[:, :, 1] = normalized_image2

    """cv2.imshow("B/G", rgb_image)
    cv2.imshow("Piezo", rgb_image[:, :, 0])
    cv2.imshow("Masked amb (G)", rgb_image[:, :, 1])
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    #print(normalized_image)
    #print(norm2color)
    

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

    return rgb_image

piezo_audio_paths = ["data/audioCh3_short_filtered.flac",
                     "data/audioCh4_short_filtered.flac"]
amb_audio_paths = ["data/audioCh1_short_filtered.flac",
                   "data/audioCh2_short_filtered.flac"]
label_paths = ["data/audioCh3_short_filtered.txt",
               "data/audioCh4_short_filtered.txt"]

"""piezo_audio_paths = ["data/Bl122.flac",
            "data/Li145.flac",
            "data/Or61.flac",
            "data/Ti81.flac"]

amb_audio_paths = ["data/Bl122_Ambient.flac",
            "data/Li145_Ambient.flac",
            "data/Or61_Ambient.flac",
            "data/Ti81_Ambient.flac"]

label_paths = ["data/Bl122.txt",
            "data/Li145.txt",
            "data/Or61.txt",
            "data/Ti81.txt"]"""

"""piezo_audio_paths = ['data/Bl122.flac']
amb_audio_paths = ['data/Bl122_Ambient.flac']

label_paths = ['data/Bl122_unseen.txt']"""

write_dir = "spec_twobird/"
df_list = []

for index, audio in enumerate(piezo_audio_paths):
    audio_name = audio.split('/')[1].split(".")[0]

    # Create Dataset
    fs = 30_000
    df = pd.read_csv(label_paths[index], 
                     sep='\t', 
                     header=None)
    
    df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
    df['type'] = df['label'].apply(lambda x: 'voc' if x == 1 else 'noise')
    print(df)
    df['onset_sample'] = (df['onset'] * fs).astype(int)
    df['offset_sample'] = (df['offset'] * fs).astype(int)
    df['length'] = df['offset_sample'] - df['onset_sample']
    
    #df['path'] = df['onset_sample'].astype(str).apply(lambda x: os.path.join(write_dir,  df['type'].astype(str), audio_name+ '_' + x + '.jpg'))
    df['path'] = df.apply(lambda row: os.path.join(write_dir, row['type'], f"{audio_name}_{row['onset_sample']}.jpg").replace('\\', '/'), axis=1)
    df['bird'] = str(audio_name)

    category_counts = df['label'].value_counts()
    print(category_counts)
    df_list.append(df)
    print(df)

    # Iterate across rows
    [y_piezo, fs] = load_filter_audio(piezo_audio_paths[index])
    [y_amb, fs] = load_filter_audio(amb_audio_paths[index])

    for index, row in df.iterrows():
        start = int(row['onset_sample'])
        end = int(row['offset_sample'])

        piezo_temp = y_piezo[start:end]
        amb_temp = y_amb[start:end]

        #temp = np.zeros(int(fs*0.255))
        #temp[0:len(piezo_temp)] = piezo_temp

        # Tiling the audio
        [mul, rem] = np.divmod(fs*0.254,len(piezo_temp))
        piezo_temp = np.tile(piezo_temp, int(mul+1))
        piezo_temp = piezo_temp[0:int(fs*0.224)]

        amb_temp = np.tile(amb_temp, int(mul+1))
        amb_temp = amb_temp[0:int(fs*0.224)]

        # Create Mel Spectrogram
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
        # Ambient
        stftMat_mel2 = lr.feature.melspectrogram(
                        y = amb_temp, 
                        sr=fs,
                        n_fft=2048, 
                        win_length=int(fs*0.008),
                        hop_length=int(fs*0.001), 
                        center=True, 
                        window='hann',
                        n_mels=225,
                        fmax=10000)
        
        # Create mask and apply binary thresholding
        abs_piezo_spec = np.abs(stftMat_mel)**2
        log_piezo = np.abs(10 * np.log10(abs_piezo_spec))
        threshold_value = 60
        _, mask = cv2.threshold(log_piezo, threshold_value, 255, cv2.THRESH_BINARY)
        mask = 255 - mask
        mask = cv2.inRange(mask, 254, 255)

        print(f"\tWriting: {row['path']}, {stftMat_mel.shape}")
        
        # Originally there was no squaring.
        normalized_image1 = spec2dB(stftMat_mel, stftMat_mel2, mask) # Piezo
        
        #cv2.imshow("mask_piezo", mask)
        #cv2.imshow("test1",colormap1)
        #cv2.imshow("test2",normalized_image1)

        cv2.imwrite(row['path'], normalized_image1)
        #cv2.imwrite(f"mask/{int(row['onset_sample'])}.jpg", mask)
        #cv2.imwrite("temp_fig/mask.jpg", mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

# Merge DFs
merged_df = pd.concat(df_list, ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Divide into fold
merged_df['fold'] = pd.cut(merged_df.index, bins=5, labels=False)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

merged_df = merged_df.sort_values(by='fold')

# Save merged df
merged_df.to_csv("meta_twobird.csv", index=False)


print(merged_df)
#### ~0.27s for 255 frames. Try on your own
# https://colab.research.google.com/github/enzokro/clck10/blob/master/_notebooks/2020-09-10-Normalizing-spectrograms-for-deep-learning.ipynb#scrollTo=wCB9ye5aEXBE