"""R3: exp032の誤りがExpert不一致から予測可能かを診断する。

固定discovery内のcross-fitted exp032/Expert予測だけを入力にし、その上でもう一段の
cross-fittingを行う。構造化94列を再投入せず、base confidenceとExpert間不一致だけを見る。

この診断はcorrection modelを作る前のgateである。error AUCが安定して0.60〜0.62を超え、
confidence-only対照も上回る場合だけR5へ進む。ここではlockboxを開かない。

  python3 exp/diagnose_residual_predictability.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import THS  # noqa: E402
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402


CACHE = Path("exp/_r1_exp032_discovery.npz")
OUT_JSON = Path("exp/_r3_residual_predictability.json")
OUT_NPZ = Path("exp/_r3_residual_predictability.npz")
N_SHUFFLES = 20
MODEL_C = 0.1


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    scores = [f1_score(y, p >= t) for t in THS]
    return float(THS[int(np.argmax(scores))])


def base_features(base: np.ndarray, experts: dict[str, np.ndarray],
                  names: tuple[str, ...]):
    matrix = np.column_stack([experts[n] for n in names])
    eps = 1e-6
    clipped = np.clip(base, eps, 1 - eps)
    entropy = -(xlogy(clipped, clipped) + xlogy(1 - clipped, 1 - clipped))
    cols = {
        "base_probability": base,
        "base_entropy": entropy,
        "expert_std": matrix.std(axis=1),
        "expert_range": matrix.max(axis=1) - matrix.min(axis=1),
        "E3_minus_E4": experts["E3_dx_text"] - experts["E4_org_text"],
        "E7_minus_E0": experts["E7_cross"] - experts["E0_anchor"],
        "E0b_minus_E0": experts["E0b_linear"] - experts["E0_anchor"],
        "E2_minus_E1": experts["E2_survey"] - experts["E1_finance"],
    }
    return np.column_stack(list(cols.values())), list(cols)


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=MODEL_C, class_weight="balanced", max_iter=3000, random_state=0,
        ),
    )


def run():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} がない。先に compare_target_aware_text.py または"
            "lockbox_error_analysis.pyでdiscovery OOFを生成すること。"
        )
    z = np.load(CACHE, allow_pickle=True)
    y = z["y"].astype(int)
    names = tuple(z["names"].tolist())
    experts = {n: z[f"oof_{n}"] for n in names}
    base = z["blend"]
    X, feature_names = base_features(base, experts, names)

    error_oof = np.full(len(y), np.nan)
    confidence_oof = np.full(len(y), np.nan)
    residual_oof = np.full(len(y), np.nan)
    shuffled_oof = np.full((N_SHUFFLES, len(y)), np.nan)
    fold_rows = []
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=LOCKBOX_SEED,
    )
    for fold, (tr, va) in enumerate(folds.split(X, y), 1):
        th = best_threshold(y[tr], base[tr])
        err_tr = ((base[tr] >= th).astype(int) != y[tr]).astype(int)
        err_va = ((base[va] >= th).astype(int) != y[va]).astype(int)
        if len(np.unique(err_tr)) < 2 or len(np.unique(err_va)) < 2:
            raise RuntimeError(f"fold {fold}でerror targetが単一class")

        # 閾値からの距離だけを使う対照。full modelはこれを含む。
        margin_tr = np.abs(base[tr] - th)[:, None]
        margin_va = np.abs(base[va] - th)[:, None]
        full_tr = np.column_stack((X[tr], margin_tr))
        full_va = np.column_stack((X[va], margin_va))
        confidence_oof[va] = classifier().fit(
            margin_tr, err_tr,
        ).predict_proba(margin_va)[:, 1]
        error_oof[va] = classifier().fit(
            full_tr, err_tr,
        ).predict_proba(full_va)[:, 1]

        residual_target = y[tr] - base[tr]
        residual_oof[va] = make_pipeline(
            StandardScaler(), Ridge(alpha=10.0),
        ).fit(full_tr, residual_target).predict(full_va)

        for s in range(N_SHUFFLES):
            rng = np.random.default_rng(LOCKBOX_SEED + fold * 1000 + s)
            perm = rng.permutation(err_tr)
            shuffled_oof[s, va] = classifier().fit(
                full_tr, perm,
            ).predict_proba(full_va)[:, 1]

        fold_rows.append({
            "fold": fold,
            "threshold": th,
            "n_error_train": int(err_tr.sum()),
            "n_error_valid": int(err_va.sum()),
            "error_auc": float(roc_auc_score(err_va, error_oof[va])),
            "confidence_auc": float(roc_auc_score(err_va, confidence_oof[va])),
        })

    assert np.isfinite(error_oof).all()
    assert np.isfinite(confidence_oof).all()
    assert np.isfinite(residual_oof).all()
    assert np.isfinite(shuffled_oof).all()

    # pooled error targetはfoldごとのtrain-only thresholdから作り直す。
    error_target = np.zeros(len(y), dtype=int)
    for (tr, va), row in zip(folds.split(X, y), fold_rows):
        error_target[va] = ((base[va] >= row["threshold"]).astype(int) != y[va])

    error_auc = float(roc_auc_score(error_target, error_oof))
    confidence_auc = float(roc_auc_score(error_target, confidence_oof))
    error_ap = float(average_precision_score(error_target, error_oof))
    shuffle_aucs = np.array([
        roc_auc_score(error_target, shuffled_oof[s]) for s in range(N_SHUFFLES)
    ])
    true_residual = y - base
    residual_r2 = float(r2_score(true_residual, residual_oof))
    residual_spearman = float(spearmanr(true_residual, residual_oof).statistic)

    result = {
        "seed": LOCKBOX_SEED,
        "n_discovery": len(y),
        "n_errors": int(error_target.sum()),
        "error_rate": float(error_target.mean()),
        "feature_names": feature_names + ["abs_margin_to_train_threshold"],
        "model": f"StandardScaler + balanced LogisticRegression(C={MODEL_C})",
        "error_auc": error_auc,
        "error_ap": error_ap,
        "confidence_only_auc": confidence_auc,
        "increment_over_confidence": error_auc - confidence_auc,
        "shuffle_auc_mean": float(shuffle_aucs.mean()),
        "shuffle_auc_std": float(shuffle_aucs.std()),
        "shuffle_auc_max": float(shuffle_aucs.max()),
        "residual_r2": residual_r2,
        "residual_spearman": residual_spearman,
        "folds": fold_rows,
        "lockbox_opened": False,
    }

    print(f"discovery={len(y)} errors={error_target.sum()} "
          f"rate={error_target.mean():.3f}")
    print("\n=== Error predictability ===")
    print(f"  full disagreement AUC : {error_auc:.4f}")
    print(f"  confidence-only AUC   : {confidence_auc:.4f}")
    print(f"  incremental AUC       : {error_auc - confidence_auc:+.4f}")
    print(f"  error AP              : {error_ap:.4f}")
    print(f"  shuffled AUC          : {shuffle_aucs.mean():.4f}±{shuffle_aucs.std():.4f} "
          f"(max {shuffle_aucs.max():.4f})")
    print("\n=== Signed residual ===")
    print(f"  cross-fitted R2       : {residual_r2:.4f}")
    print(f"  Spearman              : {residual_spearman:.4f}")
    print("\n=== Fold AUC ===")
    for row in fold_rows:
        print(f"  fold {row['fold']}: full {row['error_auc']:.4f} / "
              f"confidence {row['confidence_auc']:.4f} / "
              f"errors {row['n_error_valid']}")

    if (error_auc >= 0.62 and error_auc > confidence_auc
            and min(r["error_auc"] for r in fold_rows) >= 0.60):
        verdict = ("PASS: R5 correctionへ進む（signed Ridgeは不発のため、"
                   "base-logit offset型だけを試す）")
    elif error_auc >= 0.60 and error_auc > confidence_auc:
        verdict = "PARK: 弱い。新表現が得られた場合だけ再診断"
    else:
        verdict = "REJECT: residual correction路線を終了"
    result["verdict"] = verdict
    print(f"\n判定: {verdict}")

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ, y=y, base=base, error_target=error_target,
        error_oof=error_oof, confidence_oof=confidence_oof,
        residual_oof=residual_oof, shuffled_oof=shuffled_oof,
    )
    print(f"保存: {OUT_JSON} / {OUT_NPZ}")


if __name__ == "__main__":
    run()
