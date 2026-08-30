"""H89: is there anything left in this feature space, inside the band?

Every attempt to improve the ambiguous band has failed (five directions, see
ambiguous_band_closed).  Before asking "what information should stage 2 add",
this asks whether the existing feature space still holds unexploited structure
there, or whether the band is irreducible given these columns.

The test is local rather than model-based, so it does not inherit any model's
inductive bias.  For each row, look at its nearest neighbours among the *other*
labelled rows and ask how often they share its label.

  * if neighbours in the band agree no more than the band's base rate, then
    companies that look alike genuinely buy differently -- the columns cannot
    separate them and no combiner or second stage will
  * if they agree well above the base rate, there is local structure the global
    models are smoothing away, and a local method is worth building

Three feature spaces are checked separately, because "looks alike" depends on
what you look at: the tabular anchor columns, the org+overview embedding, and
the DX outlook embedding.

Reported alongside: the blend's own AUC inside the band, so the neighbour
agreement can be read against how much the current model already extracts.

Run:
  .venv/bin/python exp/diagnose_band_local_structure.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import _inputs  # noqa: E402
from diagnose_meta_alpha import CONFIGS, EXPERTS, _load  # noqa: E402
from embedding_features import CONCAT_SLUGS, load_concat_embeddings, load_embeddings  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha_1se  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

BAND_LO, BAND_HI = 0.1, 0.9
NEIGHBOURS = (1, 5, 10)


def _blend(y, Z, seed):
    out = np.zeros(len(y))
    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        alpha = select_alpha_1se(Z[tr], y[tr], seed)
        w, b = fit_meta(Z[tr], y[tr], alpha=alpha)
        out[va] = apply_meta(Z[va], w, b)
    return out


def _agreement(features, y, mask, k, pool=None):
    """mask の行について、自分以外の最近傍k件がラベルを共有する割合。

    pool を渡すとその行集合の中だけから近傍を探す（領域内近傍）。
    """
    pool = np.ones(len(y), bool) if pool is None else pool
    index = np.flatnonzero(pool)
    model = NearestNeighbors(
        n_neighbors=min(k + 1, len(index)), metric="cosine"
    ).fit(features[pool])
    _, idx = model.kneighbors(features[mask])
    neighbours = index[idx][:, 1:]               # 自分自身を外す
    return float((y[neighbours] == y[mask][:, None]).mean())


def _chance(y_region, y_pool):
    """近傍を y_pool からランダムに引いたときにラベルが一致する確率。

    近傍が全742行から来るなら帰無値は全体の購入率で決まる。領域内の購入率で
    計算すると、確信帯が『偶然以下』という無意味な値になる。
    """
    rate = float(np.mean(y_pool))
    return float(np.mean(y_region * rate + (1 - y_region) * (1 - rate)))


def main(config, seed):
    X, y_ref, _, _, _, columns = _inputs()
    y, Z, _ = _load(CONFIGS[config], seed)
    assert np.array_equal(y, y_ref)
    p = _blend(y, Z, seed)

    regions = {
        "確信-非購入 (p<0.1)": p < BAND_LO,
        "曖昧帯 (0.1-0.9)": (p >= BAND_LO) & (p <= BAND_HI),
        "確信-購入 (p>0.9)": p > BAND_HI,
    }

    tabular = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ).fit_transform(X[columns[0]].select_dtypes(include=[np.number]))
    concat_embed, _ = load_concat_embeddings(CONCAT_SLUGS)
    dx_embed, _ = load_embeddings("dx_outlook")
    spaces = {"表形式(anchor列)": tabular,
              "企業概要+組織図の埋め込み": concat_embed,
              "DX展望の埋め込み": dx_embed}

    print(f"seed={seed}   合成の帯内AUC: ", end="")
    band = regions["曖昧帯 (0.1-0.9)"]
    print(f"{roc_auc_score(y[band], p[band]):.4f}   帯の行数 {int(band.sum())}"
          f"   帯の実購入率 {y[band].mean():.4f}")

    rows = []
    for region, mask in regions.items():
        if mask.sum() < 20:
            continue
        for space, features in spaces.items():
            entry = {"領域": region, "n": int(mask.sum()), "特徴空間": space,
                     "偶然の一致率": _chance(y[mask], y)}
            for k in NEIGHBOURS:
                entry[f"最近傍{k}件の一致率"] = _agreement(features, y, mask, k)
            rows.append(entry)
            # 領域の中だけで近傍を探す版。「似た曖昧企業は同じ判断をするか」
            within = {"領域": f"{region} [領域内近傍]", "n": int(mask.sum()),
                      "特徴空間": space,
                      "偶然の一致率": _chance(y[mask], y[mask])}
            for k in NEIGHBOURS:
                within[f"最近傍{k}件の一致率"] = _agreement(
                    features, y, mask, k, pool=mask)
            rows.append(within)
    frame = pd.DataFrame(rows)
    for k in NEIGHBOURS:
        frame[f"超過{k}"] = frame[f"最近傍{k}件の一致率"] - frame["偶然の一致率"]

    print("\n=== 近傍のラベル一致率（自分自身は除く）===")
    show = (["領域", "n", "特徴空間", "偶然の一致率"]
            + [f"最近傍{k}件の一致率" for k in NEIGHBOURS]
            + [f"超過{k}" for k in NEIGHBOURS])
    print(frame[show].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    frame.to_csv("exp/band_local_structure.csv", index=False)
    print("\nsaved: exp/band_local_structure.csv")
    print("\n読み方: 『超過』が0付近なら、その空間で似ている企業は"
          "ラベルを共有していない＝その列からは分離できない。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config, args.seed)
