"""外部調達理由6列の独立LR expertを現行H2-both4へ追加比較する。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import _scores, blend_oof  # noqa: E402
from external_sourcing_reason_features import (  # noqa: E402
    COLS, make_external_sourcing_reason_features,
)

CACHE = "exp/_experts_seed42_concatemb_crosslr_lin0.03_llmcore4e0onlyh2both4.npz"
NAME = "E_external_reason"
SEEDS = (42, 0, 1, 2, 3)


def candidate(seed, x, y, xt):
    oof, tests = np.zeros(len(y)), []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in cv.split(x, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, solver="liblinear", max_iter=3000, random_state=seed))
        model.fit(x.iloc[tr], y[tr])
        oof[va] = model.predict_proba(x.iloc[va])[:, 1]
        tests.append(model.predict_proba(xt)[:, 1])
    return oof, np.mean(tests, axis=0)


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    x = make_external_sourcing_reason_features(train)
    xt = make_external_sourcing_reason_features(test)
    stability = pd.DataFrame({"train_rate": x.mean(), "test_rate": xt.mean(),
                              "train_count": x.sum(), "test_count": xt.sum()})
    stability.to_csv("exp/result_external_sourcing_reason_distribution.csv")
    print("=== label-free distribution ===\n", stability, flush=True)

    d = np.load(CACHE)
    y = d["y"].astype(int)
    base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
    base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
    rows = []
    for seed in SEEDS:
        p, q = candidate(seed, x, y, xt)
        oo, tt = {**base_oof, NAME: p}, {**base_te, NAME: q}
        base, _, _, _ = blend_oof(y, oo, tt, seed, experts=BASE_EXPERTS)
        aug, _, w, _ = blend_oof(y, oo, tt, seed, experts=BASE_EXPERTS + (NAME,))
        sb, sc, ss = _scores(y, base), _scores(y, aug), _scores(y, p)
        rows.append({"seed": seed, "base": sb, "candidate": sc, "standalone": ss,
                     "delta": {m: sc[m] - sb[m] for m in ("auc", "ap", "f1")},
                     "weight": float(w[-1]),
                     "max_abs_corr": float(max(abs(spearmanr(p, base_oof[n]).statistic)
                                               for n in BASE_EXPERTS))})
        print(seed, rows[-1], flush=True)
    mean = {m: float(np.mean([r["delta"][m] for r in rows]))
            for m in ("auc", "ap", "f1")}
    wins = {m: int(sum(r["delta"][m] > 0 for r in rows))
            for m in ("auc", "ap", "f1")}
    result = {"features": COLS, "rows": rows, "mean_delta": mean, "wins": wins,
              "mean_weight": float(np.mean([r["weight"] for r in rows])),
              "verdict": "PROMISING" if mean["ap"] > 0 and wins["ap"] >= 4 else "REJECT"}
    with open("exp/result_external_sourcing_reason.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
