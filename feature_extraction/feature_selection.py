import numpy as np
import torch
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph


SEED = 42
LDA_DIM = 3
PCA_DIM = 3
K_NEIGHBORS = 5

def feature_selection(samples, train_idx, test_idx, save_dir="models"):
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    # 1) 节点特征标准化
    node_scaler = NodeFeatureScaler()
    node_scaler.fit([s["node_feat"] for s in train_samples])

    # 2) PCA + LDA 选择辅助特征
    selector = PCALDASelector(pca_dim=PCA_DIM, lda_dim=LDA_DIM)

    aux_train = np.stack([s["aux_feat"] for s in train_samples], axis=0)
    y_train = np.array([s["label"] for s in train_samples])
    aux_test = np.stack([s["aux_feat"] for s in test_samples], axis=0)

    aux_train_selected = selector.fit_transform(aux_train, y_train)#得到降维后的数据
    aux_test_selected = selector.transform(aux_test)

    #  保存处理器
    joblib.dump(node_scaler, os.path.join(save_dir, "node_scaler.pkl"))
    joblib.dump(selector, os.path.join(save_dir, "selector.pkl"))

    print("✅ 已保存：node_scaler.pkl 和 selector.pkl")

    # 3) 构图
    train_graphs = []
    for s, aux_selected in zip(train_samples, aux_train_selected):
        node_feat_scaled = node_scaler.transform(s["node_feat"])
        graph = build_graph(node_feat_scaled, aux_selected, s["label"])
        train_graphs.append(graph)

    test_graphs = []
    for s, aux_selected in zip(test_samples, aux_test_selected):
        node_feat_scaled = node_scaler.transform(s["node_feat"])
        graph = build_graph(node_feat_scaled, aux_selected, s["label"])
        test_graphs.append(graph)

    return train_graphs, test_graphs, selector.output_dim


class PCALDASelector:
    def __init__(self, pca_dim=PCA_DIM, lda_dim=LDA_DIM):
        self.pca_dim = pca_dim
        self.lda_dim = lda_dim
        self.scaler = StandardScaler()
        self.pca = None
        self.lda = None
        self.output_dim = None

    def fit(self, X, y):#X: 特征 (n_samples, n_features),y: 标签，训练模型
        X_std = self.scaler.fit_transform(X)#标准化数据

        pca_dim = min(self.pca_dim, X_std.shape[1], X_std.shape[0])
        self.pca = PCA(n_components=pca_dim, random_state=SEED)#创建PCA模型
        X_pca = self.pca.fit_transform(X_std)#利用PCA降维数据

        n_classes = len(np.unique(y))
        lda_dim = min(self.lda_dim, n_classes - 1, X_pca.shape[1])

        if lda_dim <= 0:
            self.lda = None
            self.output_dim = X_pca.shape[1]
        else:
            self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)#创建LDA模型
            self.lda.fit(X_pca, y)#训练LDA模型
            self.output_dim = lda_dim

    def transform(self, X):
        X_std = self.scaler.transform(X)
        X_pca = self.pca.transform(X_std)
        if self.lda is not None:
            return self.lda.transform(X_pca).astype(np.float32)
        return X_pca.astype(np.float32)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



class NodeFeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, node_feature_list):
        X = np.concatenate(node_feature_list, axis=0)
        self.scaler.fit(X)#计算均值与标准差

    def transform(self, node_features):
        return self.scaler.transform(node_features).astype(np.float32)

def build_graph(node_features, aux_features, label):
    x = torch.tensor(node_features, dtype=torch.float)#把数组变成 PyTorch张量

    # KNN图
    n_neighbors = min(K_NEIGHBORS, max(1, len(node_features) - 1))
    if len(node_features) <= 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        A = kneighbors_graph(node_features, n_neighbors=n_neighbors, mode='connectivity')#将返回带1和0的连通性矩阵
        edge_index = np.array(A.nonzero())
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long)
    z = torch.tensor(aux_features, dtype=torch.float).unsqueeze(0)  # [1, aux_dim]

    data = Data(x=x, edge_index=edge_index, y=y, z=z)
    return data
