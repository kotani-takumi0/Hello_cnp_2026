"""H31: E0b と同じ94列を低自由度の加法モデルで読む。

E0b(H28) は one-hot + 標準化 + 線形LR、E0 は LightGBM なので、両者の間にある
「各特徴の非線形な形は読めるが、自由な高次交互作用は作らない」モデルが空いている。
連続値・順序尺度を分位点スプライン、カテゴリ3列を one-hot にし、強いL2付きLRで
合成する。入力は E0b と完全に同じ（fold内で作ったDX展望・組織図確率を含む）。

設定は H31 の事前スクリーニングで固定:
  n_knots=3 / degree=2 / knots="quantile" / C=0.01

学習foldだけで中央値・分位点・カテゴリ水準をfitする sklearn Pipeline なので、
valid/test の分布やラベルは前処理に入らない。
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

from make_submission_h1_h8 import CAT_COLS

GAM_C = 0.01
GAM_N_KNOTS = 3
GAM_DEGREE = 2
GAM_LR = dict(max_iter=5000, random_state=0)


def build_gam(columns, C=GAM_C):
    """列名から、fold内fitされる低自由度GAMパイプラインを返す。"""
    cats = [c for c in CAT_COLS if c in columns]
    nums = [c for c in columns if c not in cats]
    pre = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median"),
            SplineTransformer(
                n_knots=GAM_N_KNOTS,
                degree=GAM_DEGREE,
                knots="quantile",
                include_bias=False,
            ),
        ), nums),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        ), cats),
    ])
    return make_pipeline(
        pre,
        StandardScaler(with_mean=False),
        LogisticRegression(C=C, **GAM_LR),
    )


def fit_gam(A_tr, ytr, A_va, A_te, C=GAM_C):
    """E0b と同じ入力規約で (valid確率, test確率) を返す。"""
    model = build_gam(list(A_tr.columns), C=C)
    model.fit(A_tr, ytr)
    return (model.predict_proba(A_va)[:, 1],
            model.predict_proba(A_te)[:, 1])
