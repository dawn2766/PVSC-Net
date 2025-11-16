import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'  # 原始数据目录
OUTPUT_DIR = 'processed'  # 保存预处理数据的文件夹
SAMPLE_RATE = 16000  # 采样率
CLIP_DURATION = 5  # 每个音频片段的时长（秒）
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION  # 每个片段的采样点数
OVERLAP_RATIO = 0.5  # 窗口重叠率50%
HOP_SAMPLES = int(CLIP_SAMPLES * (1 - OVERLAP_RATIO))  # 移动步长
N_MELS = 64  # 梅尔频谱的频带数

def extract_mel(audio, sr):
    # 提取梅尔频谱，并转为dB刻度
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def load_and_split_audio(file_path):
    """
    加载音频文件，使用移动窗口法切分为多个片段
    窗口重叠率为50%，即步长为窗口长度的50%
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        clips: 音频片段列表
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(audio)
    clips = []
    
    # 使用移动窗口：起始位置以HOP_SAMPLES为步长移动
    start = 0
    while start + CLIP_SAMPLES <= total_samples:
        clip = audio[start:start + CLIP_SAMPLES]
        clips.append(clip)
        start += HOP_SAMPLES  # 移动50%窗口长度
    
    # 如果剩余部分足够长（至少窗口长度的50%），也添加进去
    if total_samples - start >= CLIP_SAMPLES * 0.5:
        # 从末尾取完整窗口
        clip = audio[-CLIP_SAMPLES:]
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
        
        print(f"处理类别: {vessel_type}")
        file_count = 0
        clip_count = 0
        
        for fname in os.listdir(vessel_dir):
            if not fname.endswith('.wav'):
                continue  # 跳过非wav文件
            fpath = os.path.join(vessel_dir, fname)
            clips = load_and_split_audio(fpath)
            file_count += 1
            clip_count += len(clips)
            
            for clip in clips:
                mel = extract_mel(clip, SAMPLE_RATE)
                X.append(mel)
                y.append(label_idx)
        
        print(f"  - 文件数: {file_count}, 生成片段数: {clip_count}")
    
    X = np.array(X)  # (样本数, 频带数, 帧数)
    y = np.array(y)
    
    print(f"\n总样本数: {len(X)}")
    print(f"特征形状: {X.shape}")
    print(f"类别数: {len(labels)}")
    print(f"类别名称: {labels}")
    
    return X, y, labels

def save_dataset():
    # 创建输出目录并保存特征、标签和类别名
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("=" * 50)
    print("开始数据预处理...")
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"片段时长: {CLIP_DURATION} 秒")
    print(f"片段采样点数: {CLIP_SAMPLES}")
    print(f"窗口重叠率: {OVERLAP_RATIO * 100}%")
    print(f"移动步长: {HOP_SAMPLES} 采样点 ({HOP_SAMPLES / SAMPLE_RATE:.2f} 秒)")
    print("=" * 50 + "\n")
    
    X, y, labels = prepare_dataset()
    
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), labels)
    
    print("\n" + "=" * 50)
    print("数据预处理完成！")
    print(f"数据已保存至: {OUTPUT_DIR}/")
    print("=" * 50)

if __name__ == '__main__':
    save_dataset()
