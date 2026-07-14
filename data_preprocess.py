import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np

DATA_DIR = 'data'  # 原始数据目录
OUTPUT_DIR = 'processed'  # 保存预处理数据的文件夹
SAMPLE_RATE = 16000  # 采样率
CLIP_DURATION = 5  # 每个音频片段的时长（秒）
OVERLAP_RATIO = 0.75  # 窗口重叠率75%
WINDOW_JITTER_RATIO = 0.2  # 内部窗口最多抖动20%的步长
RANDOM_SEED = 2026
N_MELS = 64  # 梅尔频谱的频带数


def extract_mel(audio, sr):
    # 提取梅尔频谱，并转为dB刻度
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def generate_window_starts(
    total_samples: int,
    clip_samples: int,
    hop_samples: int,
    jitter_samples: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    if total_samples < clip_samples:
        return np.empty(0, dtype=np.int64)

    max_start = total_samples - clip_samples
    starts = np.arange(0, max_start + 1, hop_samples, dtype=np.int64)
    if starts.size == 0 or starts[-1] != max_start:
        starts = np.append(starts, max_start)

    if jitter_samples > 0 and starts.size > 2:
        offsets = random_generator.integers(
            -jitter_samples,
            jitter_samples + 1,
            size=starts.size - 2,
        )
        starts[1:-1] = np.clip(starts[1:-1] + offsets, 1, max_start - 1)

    return np.unique(starts)


def load_and_split_audio(
    file_path,
    clip_samples: int,
    hop_samples: int,
    jitter_samples: int,
    random_generator: np.random.Generator,
):
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
    starts = generate_window_starts(
        total_samples,
        clip_samples,
        hop_samples,
        jitter_samples,
        random_generator,
    )
    clips = [audio[start:start + clip_samples] for start in starts]
    return clips, starts


def prepare_dataset(
    data_dir: Path,
    clip_samples: int,
    hop_samples: int,
    jitter_samples: int,
    seed: int,
):
    # 遍历所有类别和音频文件，生成特征和标签
    X, y, groups, window_starts, labels = [], [], [], [], []
    random_generator = np.random.default_rng(seed)
    for label_idx, vessel_type in enumerate(sorted(os.listdir(data_dir))):
        vessel_dir = data_dir / vessel_type
        if not vessel_dir.is_dir():
            continue  # 跳过非文件夹
        labels.append(vessel_type)
        
        print(f"处理类别: {vessel_type}")
        file_count = 0
        clip_count = 0
        
        for fname in sorted(os.listdir(vessel_dir)):
            if not fname.lower().endswith('.wav'):
                continue  # 跳过非wav文件
            fpath = vessel_dir / fname
            clips, starts = load_and_split_audio(
                fpath,
                clip_samples,
                hop_samples,
                jitter_samples,
                random_generator,
            )
            file_count += 1
            clip_count += len(clips)
            
            for clip, start in zip(clips, starts):
                mel = extract_mel(clip, SAMPLE_RATE)
                X.append(mel)
                y.append(label_idx)
                groups.append(f"{vessel_type}/{fname}")
                window_starts.append(start)
        
        print(f"  - 文件数: {file_count}, 生成片段数: {clip_count}")
    
    X = np.array(X)  # (样本数, 频带数, 帧数)
    y = np.array(y)
    groups = np.array(groups)
    window_starts = np.asarray(window_starts, dtype=np.int64)
    
    print(f"\n总样本数: {len(X)}")
    print(f"特征形状: {X.shape}")
    print(f"类别数: {len(labels)}")
    print(f"类别名称: {labels}")
    
    return X, y, groups, window_starts, labels


def save_dataset(
    data_dir: Path,
    output_dir: Path,
    clip_duration: float,
    overlap_ratio: float,
    jitter_ratio: float,
    seed: int,
):
    if clip_duration <= 0:
        raise ValueError("clip_duration must be positive")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if not 0 <= jitter_ratio < 0.5:
        raise ValueError("jitter_ratio must be in [0, 0.5)")

    clip_samples = int(round(SAMPLE_RATE * clip_duration))
    hop_samples = max(1, int(round(clip_samples * (1 - overlap_ratio))))
    jitter_samples = int(round(hop_samples * jitter_ratio))

    # 创建输出目录并保存特征、标签和类别名
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("开始数据预处理...")
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"片段时长: {clip_duration} 秒")
    print(f"片段采样点数: {clip_samples}")
    print(f"窗口重叠率: {overlap_ratio * 100}%")
    print(f"移动步长: {hop_samples} 采样点 ({hop_samples / SAMPLE_RATE:.2f} 秒)")
    print(f"窗口抖动: ±{jitter_samples} 采样点")
    print(f"随机种子: {seed}")
    print("=" * 50 + "\n")
    
    X, y, groups, window_starts, labels = prepare_dataset(
        data_dir,
        clip_samples,
        hop_samples,
        jitter_samples,
        seed,
    )
    
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    np.save(output_dir / 'groups.npy', groups)
    np.save(output_dir / 'window_starts.npy', window_starts)
    np.save(output_dir / 'labels.npy', labels)
    config = {
        "sample_rate": SAMPLE_RATE,
        "clip_duration": clip_duration,
        "clip_samples": clip_samples,
        "overlap_ratio": overlap_ratio,
        "hop_samples": hop_samples,
        "jitter_ratio": jitter_ratio,
        "jitter_samples": jitter_samples,
        "seed": seed,
        "num_source_files": int(len(np.unique(groups))),
        "num_samples": int(len(X)),
    }
    (output_dir / "preprocess_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    print("\n" + "=" * 50)
    print("数据预处理完成！")
    print(f"数据已保存至: {output_dir}/")
    print("=" * 50)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create grouped Mel-spectrogram windows.")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--clip-duration", type=float, default=CLIP_DURATION)
    parser.add_argument("--overlap-ratio", type=float, default=OVERLAP_RATIO)
    parser.add_argument("--jitter-ratio", type=float, default=WINDOW_JITTER_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


if __name__ == '__main__':
    args = create_argument_parser().parse_args()
    save_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        clip_duration=args.clip_duration,
        overlap_ratio=args.overlap_ratio,
        jitter_ratio=args.jitter_ratio,
        seed=args.seed,
    )
