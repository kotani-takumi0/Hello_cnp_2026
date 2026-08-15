"""H30: 3文書の共同・関係embedding表現を軽量に比較する。

完全ホールドアウトへ進む前のスクリーニング専用。既存exp032のOOFキャッシュを
基準に、seed 1本・5foldだけで以下を確認する。

1. 候補単体に信号があるか
2. 現行8エキスパートとの最大Spearman相関が0.90未満か
3. 9本目として加えたとき、合成OOFが動くか

提出ファイルや実験ログは更新しない。
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

from embedding_features import load_embeddings  # noqa: E402
from ensemble_experts import _scores, blend_oof  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(
    EXP_DIR, "_experts_seed42_concatemb_crosslr_lin0.03.npz"
)
THS = np.arange(0.05, 0.95, 0.005)


def _row_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def build_representations(dim):
    """train/testそれぞれについてH30候補の行列を作る。"""
    blocks = {
        slug: load_embeddings(slug, dim=dim)
        for slug in ("org", "overview", "dx_outlook")
    }
    out = {}
    for split in range(2):
        org = blocks["org"][split]
        overview = blocks["overview"][split]
        dx = blocks["dx_outlook"][split]

        joint = np.hstack((org, overview, dx))
        product = np.hstack((org * dx, overview * dx, org * overview))
        absdiff = np.hstack((np.abs(dx - org), np.abs(dx - overview),
                             np.abs(org - overview)))

        reps = {
            "H30_joint": joint,
            "H30_product": product,
            "H30_absdiff": absdiff,
            "H30_joint_relation": np.hstack((joint, product, absdiff)),
        }
        for name, matrix in reps.items():
            out.setdefault(name, [None, None])[split] = _row_normalize(matrix)
    return {name: tuple(pair) for name, pair in out.items()}


def fit_oof_and_test(x, x_test, y, seed, c):
    """同一5foldでOOFを作り、全学習データfitのtest予測も返す。"""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    params = dict(C=c, solver="lbfgs", max_iter=3000, random_state=0)
    for tr, va in skf.split(x, y):
        model = LogisticRegression(**params).fit(x[tr], y[tr])
        oof[va] = model.predict_proba(x[va])[:, 1]

    full = LogisticRegression(**params).fit(x, y)
    test_pred = full.predict_proba(x_test)[:, 1]
    return oof, test_pred


def standalone_scores(y, pred):
    f1s = [f1_score(y, pred >= t) for t in THS]
    best = int(np.argmax(f1s))
    return {
        "auc": roc_auc_score(y, pred),
        "ap": average_precision_score(y, pred),
        "f1": f1s[best],
        "th": THS[best],
    }


def load_current(cache):
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"{cache} が無い。先にexp032のエキスパートキャッシュを作成すること"
        )
    data = np.load(cache)
    names = [key[4:] for key in data.files if key.startswith("oof_")]
    y = data["y"]
    oof = {name: data[f"oof_{name}"] for name in names}
    test = {name: data[f"te_{name}"] for name in names}
    return y, names, oof, test


def run(dim, c, seed, cache):
    y, names, current_oof, current_test = load_current(cache)
    reps = build_representations(dim)

    candidate_oof = {}
    candidate_test = {}
    rows = []
    for name, (x, x_test) in reps.items():
        pred, test_pred = fit_oof_and_test(x, x_test, y, seed, c)
        candidate_oof[name] = pred
        candidate_test[name] = test_pred

        score = standalone_scores(y, pred)
        corr = {
            expert: abs(spearmanr(pred, current_oof[expert]).statistic)
            for expert in names
        }
        nearest = max(corr, key=corr.get)
        rows.append({
            "candidate": name,
            "n_features": x.shape[1],
            **score,
            "max_abs_r": corr[nearest],
            "nearest_expert": nearest,
        })

    table = pd.DataFrame(rows).set_index("candidate")
    print(f"H30 quick screen: seed={seed}, 5fold, dim/block={dim}, C={c}")
    print(f"baseline cache: {cache}")
    print("\n=== 候補単体 ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    base_blend, _, _, _ = blend_oof(
        y, current_oof, current_test, seed, alpha=None, experts=tuple(names)
    )
    base_score = _scores(y, base_blend)

    meta_rows = []
    for name in reps:
        oof = {**current_oof, name: candidate_oof[name]}
        test = {**current_test, name: candidate_test[name]}
        blend, _, weights, _ = blend_oof(
            y, oof, test, seed, alpha=None, experts=tuple(names + [name])
        )
        score = _scores(y, blend)
        meta_rows.append({
            "candidate": name,
            **score,
            "delta_auc": score["auc"] - base_score["auc"],
            "delta_ap": score["ap"] - base_score["ap"],
            "delta_f1": score["f1"] - base_score["f1"],
            "meta_weight": weights[-1],
        })

    meta = pd.DataFrame(meta_rows).set_index("candidate")
    print("\n=== 現行8本メタへの1本追加（クイックOOF）===")
    print("baseline:", " / ".join(
        f"{key.upper()} {base_score[key]:.4f}" for key in ("auc", "ap", "f1")
    ))
    print(meta.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\n注意: seed 1本の下見なので、採否判断や閾値選択には使わない。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    args = parser.parse_args()
    run(args.dim, args.c, args.seed, args.cache)
