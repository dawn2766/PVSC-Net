import numpy as np
import librosa

SAMPLE_RATE = 48000
DURATION = 5
N_MELS = 64
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 512

# 节点特征：Mel + MFCC
def extract_node_features(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features = np.concatenate([mel, mfcc], axis=0)
    return features.T.astype(np.float32)  # [time, feature]
