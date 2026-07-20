"""
轻量的特征提取包，源自原始 FeatureExtract/feature_extraction
已迁移到项目根下以避免对外部目录的依赖。
"""

from .features_aux import (
    AUXILIARY_FEATURE_DIM,
    AUXILIARY_FEATURE_NAMES,
    build_spectrogram_image,
    extract_aux_features,
)
from .features_node import extract_node_features
from .feature_selection import PCALDASelector, feature_selection


def load_raw_samples(*args, **kwargs):
    from .feature_extract import load_raw_samples as load_samples

    return load_samples(*args, **kwargs)

__all__ = [
    "load_raw_samples",
    "extract_aux_features",
    "build_spectrogram_image",
    "extract_node_features",
    "feature_selection",
    "PCALDASelector",
    "AUXILIARY_FEATURE_DIM",
    "AUXILIARY_FEATURE_NAMES",
]
