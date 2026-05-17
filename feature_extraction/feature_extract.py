import os
import glob

from .features_aux  import extract_aux_features
from tools.audio_process import load_audio,split_audio
from .features_node import extract_node_features

DURATION = 5
# 默认指向仓库内 data 目录
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# 原始特征提取
def load_raw_samples(data_dir=None):
    """
    遍历 data 目录并提取节点与辅助特征。

    Args:
        data_dir: 可选，数据根目录；默认使用包内的 data/。

    Returns:
        samples, classes
    """
    if data_dir is None:
        data_dir = DATA_DIR

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []

    for c in classes:
        files = glob.glob(os.path.join(data_dir, c, "*.wav"))

        for f in files:
            y = load_audio(f)
            audio_slices = split_audio(y, slice_duration=DURATION)

            for slice_y in audio_slices:
                node_feat = extract_node_features(slice_y)
                aux_feat = extract_aux_features(slice_y)

                samples.append({
                    "node_feat": node_feat,
                    "aux_feat": aux_feat,
                    "label": class_to_idx[c]
                })

    return samples, classes
