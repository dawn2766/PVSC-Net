import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'  # 原始数据目录
OUTPUT_DIR = 'processed'  # 保存预处理数据的文件夹
SAMPLE_RATE = 16000  # 采样率
CLIP_DURATION = 5  # 每个音频片段的时长（秒）
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION  # 每个片段的采样点数
N_MELS = 64  # 梅尔频谱的频带数

def extract_mel(audio, sr):
    # 提取梅尔频谱，并转为dB刻度
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def load_and_split_audio(file_path):
    # 加载音频文件，按固定长度切分为多个片段
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(audio)
    clips = []
    for start in range(0, total_samples, CLIP_SAMPLES):
        end = start + CLIP_SAMPLES
        if end > total_samples:
            break  # 不足一段长度则丢弃
        clip = audio[start:end]
        clips.append(clip)
    return clips

def prepare_dataset():
    # 遍历所有类别和音频文件，生成特征和标签
    X, y, labels = [], [], []
    for label_idx, vessel_type in enumerate(sorted(os.listdir(DATA_DIR))):
        vessel_dir = os.path.join(DATA_DIR, vessel_type)
        if not os.path.isdir(vessel_dir):
            continue  # 跳过非文件夹
        labels.append(vessel_type)
        for fname in os.listdir(vessel_dir):
            if not fname.endswith('.wav'):
                continue  # 跳过非wav文件
            fpath = os.path.join(vessel_dir, fname)
            clips = load_and_split_audio(fpath)
            for clip in clips:
                mel = extract_mel(clip, SAMPLE_RATE)
                X.append(mel)
                y.append(label_idx)
    X = np.array(X)  # (样本数, 频带数, 帧数)
    y = np.array(y)
    return X, y, labels

def save_dataset():
    # 创建输出目录并保存特征、标签和类别名
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    X, y, labels = prepare_dataset()
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), labels)

if __name__ == '__main__':
    save_dataset()
