import numpy as np
import librosa

from scipy.signal import butter, filtfilt, hilbert
from .features_glcm import extract_glcm_features
from .features_glcm import extract_hu_moments

SPEC_IMG_H = 128
SPEC_IMG_W = 128
GLCM_LEVELS = 16
PCA_DIM = 3
SAMPLE_RATE = 48000
N_FFT = 1024
HOP_LENGTH = 512
LOWCUT = 200
HIGHCUT = 2000

def extract_aux_features(y):
    """
    原始辅助特征 = 频谱图图像特征 + 包络特征 + 基本统计
    """
    img_float, img_quant = build_spectrogram_image(y)

    glcm_feat = extract_glcm_features(img_quant)      # 14维，灰度共生矩阵
    hu_feat = extract_hu_moments(img_float)           # 7维,img_float 先用于求图像矩,拼成hu,做对数变换返回 hu_feat
    grad_feat = extract_gradient_features(img_float)  # 4维,梯度特征，均值,标准差,最大值,熵
    env_feat = extract_envelope_features(y)           # 4维，包络特征
    basic_feat = extract_basic_audio_stats(y)         # 4维，基础特征

    aux = np.concatenate(
        [glcm_feat, hu_feat, grad_feat, env_feat, basic_feat],
        axis=0
    ).astype(np.float32)

    return aux  # 共 33 维


def build_spectrogram_image(y):
    """
    构建归一化频谱图，返回 float 图像和量化图像
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S + 1e-8, ref=np.max)

    S_img = resize_2d_array(S_db, SPEC_IMG_H, SPEC_IMG_W)

    S_min = np.min(S_img)
    S_max = np.max(S_img)
    if S_max - S_min < 1e-12:
        img_float = np.zeros_like(S_img, dtype=np.float32)
    else:
        img_float = ((S_img - S_min) / (S_max - S_min)).astype(np.float32)

    img_quant = np.clip((img_float * (GLCM_LEVELS - 1)).astype(np.int32), 0, GLCM_LEVELS - 1)

    return img_float, img_quant


def extract_gradient_features(img_float):
    gy, gx = np.gradient(img_float)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    feats = np.array([
        float(np.mean(mag)),  # 均值
        float(np.std(mag)),  # 标准差
        float(np.max(mag)),  # 最大值
        float(spectral_entropy(mag.flatten()))  # 熵
    ], dtype=np.float32)
    return feats


def extract_envelope_features(y):
    y_band = bandpass_filter(y)
    env = np.abs(hilbert(y_band)).astype(np.float32)

    spectrum = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / SAMPLE_RATE)

    idx = np.where((freqs >= 0) & (freqs <= 200))[0]
    spectrum = spectrum[idx]
    freqs = freqs[idx]

    if len(spectrum) == 0 or np.sum(spectrum) <= 1e-12:
        return np.zeros(4, dtype=np.float32)

    peak_freq = freqs[np.argmax(spectrum)]
    peak_val = np.max(spectrum)
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
    entropy = spectral_entropy(spectrum)

    return np.array([peak_freq, peak_val, centroid, entropy], dtype=np.float32)


def extract_basic_audio_stats(y):
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)))
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)))
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)))

    return np.array([rms, zcr, spec_centroid, spec_bw], dtype=np.float32)


def resize_2d_array(arr, out_h=SPEC_IMG_H, out_w=SPEC_IMG_W):
    """
    用 numpy 线性插值把二维数组缩放到固定大小
    """
    in_h, in_w = arr.shape
    row_idx = np.linspace(0, in_h - 1, out_h)
    col_idx = np.linspace(0, in_w - 1, out_w)

    tmp = np.array([np.interp(col_idx, np.arange(in_w), row) for row in arr])
    out = np.array([np.interp(row_idx, np.arange(in_h), tmp[:, j]) for j in range(tmp.shape[1])]).T
    return out


def bandpass_filter(y, sr=SAMPLE_RATE, lowcut=LOWCUT, highcut=HIGHCUT, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y_filtered = filtfilt(b, a, y)
    return y_filtered.astype(np.float32)


def spectral_entropy(magnitude):
    psd = magnitude.astype(np.float64) ** 2
    psd_sum = np.sum(psd)
    if psd_sum <= 1e-12:
        return 0.0
    psd = psd / psd_sum
    psd = psd + 1e-12
    return float(-np.sum(psd * np.log(psd)))
