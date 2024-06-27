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
def spec2dB(spec, show_img = False):
    db = lr.amplitude_to_db(np.abs(spec), ref=np.max)
    normalized_image = cv2.normalize(db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normalize
    normalized_image = cv2.rotate(normalized_image, cv2.ROTATE_180) # Rotate 180 degrees
    normalized_image = cv2.flip(normalized_image, 1) # Flip horizontally

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    normalized_image = clahe.apply(normalized_image)

    if show_img == True:
        cv2.imshow("piezo", normalized_image)#, colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return normalized_image#, colormap

audio_paths = ['data/Bl122.flac']
label_paths = ['data/Bl122.txt']

df_list = []

for index, audio in enumerate(audio_paths):
    audio_name = audio.split('/')[1].split(".")[0]

    # Create Dataset
    fs = 30_000
    df = pd.read_csv(label_paths[index], 
                     sep='\t', 
                     header=None)
    df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
    df = df[df['label'] == 1]
    df['type'] = df['label'].apply(lambda x: 'voc' if x == 1 else 'noise')
    print(df)
    df['onset_sample'] = (df['onset'] * fs).astype(int)
    df['offset_sample'] = (df['offset'] * fs).astype(int)
    df['length'] = df['offset_sample'] - df['onset_sample']
    
    #df['path'] = df['onset_sample'].astype(str).apply(lambda x: os.path.join(write_dir,  df['type'].astype(str), audio_name+ '_' + x + '.jpg'))
    #df['path'] = df.apply(lambda row: os.path.join(write_dir, row['type'], f"{audio_name}_{row['onset_sample']}.jpg").replace('\\', '/'), axis=1)
    df['bird'] = str(audio_name)

    category_counts = df['label'].value_counts()
    print(category_counts)
    df_list.append(df)
    print(df)

    # Iterate across rows
    [y_piezo, fs] = load_filter_audio(audio_paths[index])
    for index, row in df.iterrows():
        start = int(row['onset_sample'])
        end = int(row['offset_sample'])

        piezo_temp = y_piezo[start:end]
        temp = np.zeros(int(fs*0.255))
        temp[0:len(piezo_temp)] = piezo_temp
        temp = temp[0:int(fs*0.224)]

        # Tiling the audio
        [mul, rem] = np.divmod(fs*0.254,len(piezo_temp))
        piezo_temp = np.tile(piezo_temp, int(mul+1))
        piezo_temp = piezo_temp[0:int(fs*0.224)]

        # Create Mel Spectrogram

        stftMat_mel_temp = lr.feature.melspectrogram(
                        y = temp, 
                        sr=fs,
                        n_fft=2048, 
                        win_length=int(fs*0.008),
                        hop_length=int(fs*0.001), 
                        center=True, 
                        window='hann',
                        n_mels=225,
                        fmax=10000)
        normalized_image1 = spec2dB(stftMat_mel_temp)
        cv2.imwrite("raw1.jpg", normalized_image1)

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
        normalized_image1 = spec2dB(stftMat_mel)
        cv2.imwrite("raw2.jpg", normalized_image1)
        #print(f"\tWriting: {row['path']}, {stftMat_mel.shape}")
        
        normalized_image1 = spec2dB(stftMat_mel) # Piezo

        input('Stop')



