"""H32: 埋め込み空間の局所近傍を使う cosine-kNN エキスパート。

既存 E3/E4 は TF-IDF/embedding にロジスティック回帰の超平面を引く。
H32 は「既知の購入企業に局所的に似ているか」を測り、モデル族として異なる
順位を作る。探索スクリーニング後、設定は次の1点に固定する。

  入力       [今後のDX展望 ; 組織図] embedding（各1024次元、等重み正規化）
  距離       cosine
  近傍数     40
  距離重み   exp(-8 * distance)
  平滑化     全体購入率20社分

`predict_proba` を持つ最小限の sklearn 互換推定器として実装し、通常OOFと
完全ホールドアウトの両方から同じコードを使う。
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from embedding_features import load_concat_embeddings

KNN_SLUGS = ("dx_outlook", "org")
KNN_DIM = 1024
KNN_NEIGHBORS = 40
KNN_DISTANCE_SCALE = 8.0
KNN_PRIOR_STRENGTH = 20.0


def load_knn_embeddings(dim=KNN_DIM):
    """H32のtrain/test入力を返す。各文書ブロックは等重みになる。"""
    return load_concat_embeddings(KNN_SLUGS, dim=dim)


class SmoothedCosineKNN:
    """距離重み付き近傍率を全体事前率へ縮小する二値分類器。"""

    def __init__(self, n_neighbors=KNN_NEIGHBORS,
                 distance_scale=KNN_DISTANCE_SCALE,
                 prior_strength=KNN_PRIOR_STRENGTH):
        self.n_neighbors = int(n_neighbors)
        self.distance_scale = float(distance_scale)
        self.prior_strength = float(prior_strength)

    def fit(self, X, y):
        self.y_ = np.asarray(y, dtype=float)
        self.prior_ = float(self.y_.mean())
        self.k_ = min(self.n_neighbors, len(self.y_))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_, metric="cosine",
                                    algorithm="brute", n_jobs=1)
        self.nn_.fit(np.asarray(X, dtype=float))
        return self

    def predict_proba(self, X):
        distance, index = self.nn_.kneighbors(np.asarray(X, dtype=float))
        weight = np.exp(-self.distance_scale * distance)
        local = ((weight * self.y_[index]).sum(axis=1)
                 / np.maximum(weight.sum(axis=1), 1e-12))
        positive = ((self.k_ * local + self.prior_strength * self.prior_)
                    / (self.k_ + self.prior_strength))
        return np.column_stack((1.0 - positive, positive))


def build_knn_model():
    return SmoothedCosineKNN()
