"""H40 支払余力3列をdiscovery側だけでexp032へ追加比較する。

lockbox_error_analysis.py と同じ固定分割を使い、lockbox側のID・ラベル・予測は
表示も保存もしない。候補は3列+LR(C=1.0)に固定する。

実行:
  python3 exp/compare_affordability.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from affordability_features import AFFORDABILITY_COLS, affordability_frame  # noqa: E402
from lockbox_error_analysis import (  # noqa: E402
    EXP032_LINEAR_C, INNER_FOLDS, LOCKBOX_SEED, LOCKBOX_SIZE,
    _cross_fitted_meta, discovery_oof, fixed_split, lockbox_fingerprint,
)
from meta_blend import apply_meta, fit_meta, to_logit  # noqa: E402


NAME = "E12_affordability"
FIXED_ALPHAS = (0.001, 0.003, 0.01, 0.03)
CACHE = Path("exp/_lockbox_discovery_exp032_oof.npz")
OUT_CSV = Path("exp/_h40_affordability_discovery.csv")
OUT_REPORT = Path("documents/h40_affordability_discovery.md")


def _score(y, p):
    thresholds = np.arange(0.05, 0.95, 0.005)
    f1s = np.array([f1_score(y, p >= t) for t in thresholds])
    i = int(np.argmax(f1s))
    return {
        "AUC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "F1": f1s[i],
        "threshold": float(thresholds[i]),
    }


def _candidate_oof(raw, discovery, y, seed):
    x = affordability_frame(raw).iloc[discovery].reset_index(drop=True)
    out = np.zeros(len(discovery), dtype=float)
    folds = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for tr, va in folds.split(x, y):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=0),
        )
        model.fit(x.iloc[tr], y[tr])
        out[va] = model.predict_proba(x.iloc[va])[:, 1]
    return out, x


def _fixed_meta(y, oof, names, seed, alpha):
    z = to_logit(np.column_stack([oof[n] for n in names]))
    out = np.zeros(len(y), dtype=float)
    weights = []
    folds = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for tr, va in folds.split(z, y):
        w, b = fit_meta(z[tr], y[tr], alpha=alpha, target="zero")
        out[va] = apply_meta(z[va], w, b)
        weights.append(w)
    return out, np.mean(weights, axis=0)


def _load_or_compute_baseline(train, discovery, fingerprint):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        cached_fp = str(d["lockbox_fingerprint"].item())
        if cached_fp != fingerprint:
            raise RuntimeError("baseline cacheと固定lockboxのfingerprintが不一致")
        names = tuple(str(v) for v in d["names"])
        y = d["y"].astype(int)
        oof = {n: d[f"oof_{n}"] for n in names}
        blend = d["blend"]
        print(f"baseline cacheを利用: {CACHE}")
        return y, oof, blend, names

    y, oof, blend, names, alphas, weights = discovery_oof(
        train, discovery, LOCKBOX_SEED,
    )
    np.savez(
        CACHE,
        lockbox_fingerprint=np.array(fingerprint), names=np.array(names),
        y=y, blend=blend, alphas=np.array(alphas), weights=weights,
        **{f"oof_{n}": oof[n] for n in names},
    )
    print(f"baseline cacheを保存: {CACHE}")
    return y, oof, blend, names


def _transition(y, base_p, cand_p):
    sb, sc = _score(y, base_p), _score(y, cand_p)
    b = (base_p >= sb["threshold"]).astype(int)
    c = (cand_p >= sc["threshold"]).astype(int)
    rows = []
    for before, after in ((0, 1), (1, 0)):
        m = (b == before) & (c == after)
        rows.append({
            "遷移": f"{before}->{after}", "件数": int(m.sum()),
            "実購入": int(y[m].sum()), "実非購入": int(m.sum() - y[m].sum()),
        })
    return pd.DataFrame(rows)


def run():
    train = pd.read_csv("data/train.csv")
    y_all = train["購入フラグ"].to_numpy(dtype=int)
    discovery, lockbox = fixed_split(y_all)
    fingerprint = lockbox_fingerprint(train.iloc[lockbox]["企業ID"].to_numpy())
    print(f"discovery={len(discovery)} / lockbox={len(lockbox)}")
    print(f"lockbox fingerprint={fingerprint}")
    print("lockboxのID・ラベル・予測は出力しません。")

    y, base_oof, base_blend, base_names = _load_or_compute_baseline(
        train, discovery, fingerprint,
    )
    affordability_oof, x = _candidate_oof(train, discovery, y, LOCKBOX_SEED)
    candidate_oof = {**base_oof, NAME: affordability_oof}
    candidate_names = base_names + (NAME,)
    candidate_blend, auto_alphas, auto_weights = _cross_fitted_meta(
        y, candidate_oof, candidate_names, LOCKBOX_SEED,
    )

    records = []
    for label, p in ((NAME, affordability_oof),
                     ("exp032_auto", base_blend),
                     ("exp032_plus_H40_auto", candidate_blend)):
        records.append({"setting": label, "alpha": "auto", **_score(y, p)})

    fixed_rows = []
    for alpha in FIXED_ALPHAS:
        pb, wb = _fixed_meta(y, base_oof, base_names, LOCKBOX_SEED, alpha)
        pc, wc = _fixed_meta(y, candidate_oof, candidate_names,
                             LOCKBOX_SEED, alpha)
        sb, sc = _score(y, pb), _score(y, pc)
        row = {
            "setting": "fixed_alpha_pair", "alpha": alpha,
            **{f"base_{k}": v for k, v in sb.items()},
            **{f"candidate_{k}": v for k, v in sc.items()},
            "delta_AUC": sc["AUC"] - sb["AUC"],
            "delta_AP": sc["AP"] - sb["AP"],
            "delta_F1": sc["F1"] - sb["F1"],
            "H40_weight": float(wc[-1]),
        }
        fixed_rows.append(row)

    scores = pd.DataFrame(records)
    fixed = pd.DataFrame(fixed_rows)
    base_score = scores.loc[scores.setting == "exp032_auto"].iloc[0]
    cand_score = scores.loc[scores.setting == "exp032_plus_H40_auto"].iloc[0]
    delta = {m: float(cand_score[m] - base_score[m]) for m in ("AUC", "AP", "F1")}

    corr_rows = []
    for name in base_names:
        corr_rows.append({
            "相手": name,
            "Spearman": spearmanr(affordability_oof, base_oof[name]).statistic,
        })
    corr_rows.append({
        "相手": "exp032_blend",
        "Spearman": spearmanr(affordability_oof, base_blend).statistic,
    })
    corr = pd.DataFrame(corr_rows).sort_values("Spearman", ascending=False)
    transitions = _transition(y, base_blend, candidate_blend)

    # autoだけ良くfixedで消えるH34型をPROMISINGにしない。
    fixed_ap_positive = int((fixed["delta_AP"] > 0).sum())
    promising = (
        delta["AP"] >= 0.002 and delta["F1"] >= -0.002
        and fixed_ap_positive >= 3
    )
    verdict = "PROMISING（次はdiscovery内multi-seed）" if promising else \
        "STOP（lockboxは開けない）"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([
        scores.assign(section="auto_or_single"),
        fixed.assign(section="fixed_alpha"),
    ], ignore_index=True, sort=False).to_csv(OUT_CSV, index=False)
    feature_stats = x.describe().T[["mean", "std", "min", "50%", "max"]]
    report = f"""# H40 教育商材支払余力3列 — discovery限定比較

結論: **{verdict}**

固定分割: discovery {len(discovery)}件 / lockbox {LOCKBOX_SIZE}件  
lockbox fingerprint: `{fingerprint}`  
lockboxのID・ラベル・予測は表示・保存していない。

## 固定した3列

1. `{AFFORDABILITY_COLS[0]}`
2. `{AFFORDABILITY_COLS[1]}`
3. `{AFFORDABILITY_COLS[2]}`

モデルは中央値補完 + StandardScaler + LogisticRegression(C=1.0)。

## 単体とauto-alpha合成

```text
{scores.to_string(index=False)}
```

exp032への追加差: AUC {delta['AUC']:+.4f} / AP {delta['AP']:+.4f} / F1 {delta['F1']:+.4f}  
候補入りメタのfold別alpha: `{auto_alphas}`  
H40平均メタ重み: `{float(auto_weights[-1]):.4f}`

## 固定alphaでの切り分け

```text
{fixed[['alpha', 'delta_AUC', 'delta_AP', 'delta_F1', 'H40_weight']].to_string(index=False)}
```

固定alphaでAPが正だった条件: {fixed_ap_positive}/{len(FIXED_ALPHAS)}。
auto-alphaの改善が候補追加によるalpha選択変化だけでないかをここで確認する。

## 既存予測との相関

```text
{corr.to_string(index=False)}
```

## 最適閾値での予測遷移

```text
{transitions.to_string(index=False)}
```

## discovery内の特徴量分布

```text
{feature_stats.to_string()}
```

この結果は、誤り分析に使ったdiscovery上の探索結果である。PROMISINGでも直接採用せず、
discovery内multi-seedで設定を変えずに再確認してから、候補群をまとめてlockboxで評価する。
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(scores.to_string(index=False))
    print("\nfixed alpha:")
    print(fixed[["alpha", "delta_AUC", "delta_AP", "delta_F1",
                 "H40_weight"]].to_string(index=False))
    print(f"\n追加差: AUC {delta['AUC']:+.4f} / AP {delta['AP']:+.4f} / F1 {delta['F1']:+.4f}")
    print(f"H40 weight={float(auto_weights[-1]):.4f}, verdict={verdict}")
    print(f"保存: {OUT_CSV}")
    print(f"保存: {OUT_REPORT}")


if __name__ == "__main__":
    run()
