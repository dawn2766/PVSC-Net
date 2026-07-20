import numpy as np

# 图像化辅助特征参数
SPEC_IMG_H = 128
SPEC_IMG_W = 128
GLCM_LEVELS = 16
PCA_DIM = 3  # 先压到 12 维
# 注意：4 分类时 LDA 最大只能得到 3 维
# 实际会自动设为 min(LDA_DIM, 类别数-1)
LDA_DIM = 3
LOWCUT = 200
HIGHCUT = 2000
# GLCM 特征
def compute_glcm(img, dx, dy, levels=GLCM_LEVELS):
    """
    手写 GLCM，避免额外依赖 skimage
    """
    h, w = img.shape
    source_y = slice(max(0, -dy), min(h, h - dy))
    source_x = slice(max(0, -dx), min(w, w - dx))
    target_y = slice(max(0, dy), min(h, h + dy))
    target_x = slice(max(0, dx), min(w, w + dx))
    source = img[source_y, source_x].astype(np.int64, copy=False).ravel()
    target = img[target_y, target_x].astype(np.int64, copy=False).ravel()
    if source.size == 0:
        return np.zeros((levels, levels), dtype=np.float64)
    counts = np.bincount(source * levels + target, minlength=levels * levels)
    return counts.reshape(levels, levels).astype(np.float64) / source.size


def glcm_features_from_matrix(P):
    levels = P.shape[0]
    i = np.arange(levels).reshape(-1, 1)
    j = np.arange(levels).reshape(1, -1)

    contrast = np.sum(((i - j) ** 2) * P)
    dissimilarity = np.sum(np.abs(i - j) * P)
    homogeneity = np.sum(P / (1.0 + (i - j) ** 2))
    asm = np.sum(P ** 2)
    energy = np.sqrt(asm)

    mu_i = np.sum(i * P)
    mu_j = np.sum(j * P)
    sigma_i = np.sqrt(np.sum(((i - mu_i) ** 2) * P))
    sigma_j = np.sqrt(np.sum(((j - mu_j) ** 2) * P))

    if sigma_i < 1e-12 or sigma_j < 1e-12:
        correlation = 0.0
    else:
        correlation = np.sum(((i - mu_i) * (j - mu_j) * P) / (sigma_i * sigma_j))

    entropy = -np.sum(P[P > 0] * np.log(P[P > 0] + 1e-12))

    return np.array([
        contrast,
        dissimilarity,
        homogeneity,
        asm,
        energy,
        correlation,
        entropy
    ], dtype=np.float32)


# 送入四个方向的 GLCM 计算：
def extract_glcm_features(img_quant):
    directions = [
        (1, 0),  # 0°
        (1, -1),  # 45°
        (0, -1),  # 90°
        (-1, -1)  # 135°
    ]

    all_feats = []
    for dx, dy in directions:
        P = compute_glcm(img_quant, dx, dy, levels=GLCM_LEVELS)
        feat = glcm_features_from_matrix(P)
        all_feats.append(feat)

    all_feats = np.array(all_feats, dtype=np.float32)

    # 取四个方向的均值和标准差
    feat_mean = np.mean(all_feats, axis=0)
    feat_std = np.std(all_feats, axis=0)


    return np.concatenate([feat_mean, feat_std], axis=0)  # 14维


# 图像矩特征
def extract_hu_moments(img_float):
    """
    手写 Hu 不变矩，返回 7 维
    """
    h, w = img_float.shape
    yy, xx = np.mgrid[0:h, 0:w]

    m00 = np.sum(img_float) + 1e-12
    m10 = np.sum(xx * img_float)
    m01 = np.sum(yy * img_float)

    x_bar = m10 / m00
    y_bar = m01 / m00

    x = xx - x_bar
    y = yy - y_bar

    mu20 = np.sum((x ** 2) * img_float)
    mu02 = np.sum((y ** 2) * img_float)
    mu11 = np.sum((x * y) * img_float)
    mu30 = np.sum((x ** 3) * img_float)
    mu03 = np.sum((y ** 3) * img_float)
    mu12 = np.sum((x * (y ** 2)) * img_float)
    mu21 = np.sum(((x ** 2) * y) * img_float)

    eta20 = mu20 / (m00 ** 2)
    eta02 = mu02 / (m00 ** 2)
    eta11 = mu11 / (m00 ** 2)
    eta30 = mu30 / (m00 ** 2.5)
    eta03 = mu03 / (m00 ** 2.5)
    eta12 = mu12 / (m00 ** 2.5)
    eta21 = mu21 / (m00 ** 2.5)

    hu1 = eta20 + eta02
    hu2 = (eta20 - eta02) ** 2 + 4 * (eta11 ** 2)
    hu3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
    hu4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
    hu5 = ((eta30 - 3 * eta12) * (eta30 + eta12) *
           ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) +
           (3 * eta21 - eta03) * (eta21 + eta03) *
           (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2))
    hu6 = ((eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) +
           4 * eta11 * (eta30 + eta12) * (eta21 + eta03))
    hu7 = ((3 * eta21 - eta03) * (eta30 + eta12) *
           ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) -
           (eta30 - 3 * eta12) * (eta21 + eta03) *
           (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2))

    hu = np.array([hu1, hu2, hu3, hu4, hu5, hu6, hu7], dtype=np.float32)
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu
