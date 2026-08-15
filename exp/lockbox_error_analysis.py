"""exp032 の誤りを discovery 側だけで観察する。

このスクリプトは、今後の特徴量探索における人間の feature-selection leakage を
減らすための入口である。train を一度だけ discovery / lockbox に固定分割し、
モデル学習も誤り分析も discovery の中だけで完結させる。

重要:
  - lockbox の企業ID、ラベル、予測値は表示も保存もしない。
  - 出力は discovery の誤分類行だけ。
  - lockbox は候補特徴量を事前固定した後に別の評価器から一度だけ開ける。
  - この分割は、過去の exp032 自体を完全な初見で評価するものではない。
    「このレポートを見て今後作る特徴量」の増分評価に使う。

実行:
  python3 exp/lockbox_error_analysis.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_features import add_cross_features  # noqa: E402
from ensemble_experts import (  # noqa: E402
    LINEAR_C, THS, build_features, e4_inputs, expert_names,
)
from expert_groups import (  # noqa: E402
    e0_anchor_cols, e1_finance_cols, e2_survey_cols, e6_manual_cols,
    e7_cross_cols,
)
from holdout_check import fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha, to_logit  # noqa: E402
from organization_features import load_org_text  # noqa: E402
from text_features import TEXT_COL  # noqa: E402


LOCKBOX_SEED = 20260815
LOCKBOX_SIZE = 260
INNER_FOLDS = 5
EXP032_LINEAR_C = 0.03
OUT_CSV = Path("exp/_lockbox_discovery_errors.csv")
OUT_REPORT = Path("documents/error_analysis_discovery.md")
OUT_PROTOCOL = Path("exp/_lockbox_protocol.json")


def fixed_split(y: np.ndarray, seed: int = LOCKBOX_SEED,
                lockbox_size: int = LOCKBOX_SIZE):
    """固定の層化分割を返す。Bの中身は呼び出し側で出力しない。"""
    idx = np.arange(len(y))
    discovery, lockbox = train_test_split(
        idx, test_size=lockbox_size, stratify=y, random_state=seed,
    )
    return np.sort(discovery), np.sort(lockbox)


def lockbox_fingerprint(ids: np.ndarray) -> str:
    """ID自体を公開せず、将来同じ集合か照合するためのfingerprint。"""
    payload = ",".join(map(str, sorted(map(int, ids)))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    scores = [f1_score(y, p >= t) for t in THS]
    return float(THS[int(np.argmax(scores))])


def _cross_fitted_meta(y: np.ndarray, expert_oof: dict[str, np.ndarray],
                       names: tuple[str, ...], seed: int):
    """discovery 内でメタ層もcross-fitし、各行の探索用OOFを返す。"""
    z = to_logit(np.column_stack([expert_oof[n] for n in names]))
    out = np.zeros(len(y), dtype=float)
    alphas = []
    weights = []
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed,
    )
    for tr, va in folds.split(z, y):
        alpha = select_alpha(z[tr], y[tr], seed=seed, target="zero")[0]
        w, b = fit_meta(z[tr], y[tr], alpha=alpha, target="zero")
        out[va] = apply_meta(z[va], w, b)
        alphas.append(float(alpha))
        weights.append(w)
    return out, alphas, np.mean(weights, axis=0)


def discovery_oof(train: pd.DataFrame, discovery: np.ndarray, seed: int):
    """exp032相当の8エキスパートをdiscovery内だけでOOF化する。"""
    # build_features/e4_inputs は行単位特徴を全行について作るが、ラベルを使わない。
    # fit は下の discovery 内indexだけで行う。
    x, y, _ = build_features(train, train, cross=True)
    txt = train[TEXT_COL].fillna("").astype(str).values
    org = load_org_text(train)
    e4_in, _, e4_model = e4_inputs(train, train, org_embed=False,
                                    concat_embed=True)
    e4 = (e4_in, e4_model)
    cols = (
        e0_anchor_cols(x.columns),
        e1_finance_cols(with_ratios=False),
        e2_survey_cols(with_step=False),
        e6_manual_cols(x.columns),
        e7_cross_cols(),
    )
    names = expert_names(cross=True, linear=True)
    y_discovery = y[discovery].astype(int)
    oof = {n: np.zeros(len(discovery), dtype=float) for n in names}

    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed,
    )
    for fold, (tr_pos, va_pos) in enumerate(
            folds.split(discovery, y_discovery), 1):
        print(f"  discovery fold {fold}/{INNER_FOLDS} ...", flush=True)
        tr_idx, va_idx = discovery[tr_pos], discovery[va_pos]
        pred = fit_predict_experts(
            x, y, txt, org, tr_idx, va_idx, seed, cols,
            e4=e4, e7_model="lr", linear_c=EXP032_LINEAR_C,
        )
        for name in names:
            oof[name][va_pos] = pred[name]

    blend, alphas, weights = _cross_fitted_meta(
        y_discovery, oof, names, seed,
    )
    return y_discovery, oof, blend, names, alphas, weights


def _clip_text(value, width=220):
    text = " ".join(str(value).replace("|", "／").split())
    return text if len(text) <= width else text[:width] + "…"


def _clip_tail_text(value, width=260):
    """DX展望は結論が末尾に置かれることが多いので末尾側を表示する。"""
    text = " ".join(str(value).replace("|", "／").split())
    return text if len(text) <= width else "…" + text[-width:]


def _markdown_table(frame: pd.DataFrame, index: bool = False,
                    float_digits: int = 3) -> str:
    """Optional dependencyのtabulateを使わずMarkdown表を作る。"""
    shown = frame.reset_index() if index else frame.copy()

    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{float_digits}f}"
        return str(value).replace("|", "／").replace("\n", " ")

    headers = [str(c) for c in shown.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _enrich(train: pd.DataFrame, discovery: np.ndarray, y: np.ndarray,
            oof: dict[str, np.ndarray], blend: np.ndarray,
            names: tuple[str, ...], threshold: float):
    frame = train.iloc[discovery].copy().reset_index(drop=True)
    pred = (blend >= threshold).astype(int)
    frame["OOF確率"] = blend
    frame["OOF予測"] = pred
    frame["誤り種別"] = np.select(
        [(y == 1) & (pred == 0), (y == 0) & (pred == 1),
         (y == 1) & (pred == 1)],
        ["FN", "FP", "TP"], default="TN",
    )
    frame["閾値からの距離"] = np.abs(blend - threshold)
    for name in names:
        frame[f"予測_{name}"] = oof[name]
    expert_mat = np.column_stack([oof[n] for n in names])
    frame["expert_std"] = expert_mat.std(axis=1)
    frame["expert_min"] = expert_mat.min(axis=1)
    frame["expert_max"] = expert_mat.max(axis=1)

    derived = add_cross_features(train, pd.DataFrame(index=train.index))
    frame["Need_不満4"] = derived.iloc[discovery]["不満4"].to_numpy()
    frame["Capacity_ソフト投資売上比_pct"] = (
        derived.iloc[discovery]["ソフト投資_売上比"].to_numpy() * 100
    )
    frame["営業利益率_pct"] = (
        train.iloc[discovery]["営業利益"].to_numpy()
        / train.iloc[discovery]["売上"].to_numpy() * 100
    )

    llm_path = Path("data/train_with_llm_3axes.csv")
    if llm_path.exists():
        llm = pd.read_csv(
            llm_path,
            usecols=["企業ID", "llm_dx_execution_score",
                     "llm_external_vendor_need", "llm_cautiousness"],
        ).set_index("企業ID")
        for col in llm.columns:
            frame[col] = frame["企業ID"].map(llm[col])
    return frame


def _mean_table(frame: pd.DataFrame, names: tuple[str, ...]):
    cols = [
        "OOF確率", "Need_不満4", "Capacity_ソフト投資売上比_pct",
        "営業利益率_pct", "従業員数", "expert_std",
        "llm_dx_execution_score", "llm_external_vendor_need",
        "llm_cautiousness",
    ] + [f"予測_{n}" for n in names]
    cols = [c for c in cols if c in frame.columns]
    means = frame.groupby("誤り種別")[cols].mean().reindex(["FN", "FP", "TP", "TN"])
    counts = frame["誤り種別"].value_counts().reindex(means.index).fillna(0).astype(int)
    means.insert(0, "件数", counts)
    return means


def _industry_table(frame: pd.DataFrame):
    tmp = frame.assign(誤り=frame["誤り種別"].isin(["FN", "FP"]).astype(int))
    out = tmp.groupby("業界").agg(
        件数=("企業ID", "size"), 誤り数=("誤り", "sum"),
        誤り率=("誤り", "mean"), 購入率=("購入フラグ", "mean"),
        平均予測=("OOF確率", "mean"),
    )
    return out.loc[out["件数"] >= 8].sort_values(["誤り率", "件数"], ascending=False)


def _case_table(frame: pd.DataFrame, kind: str, n=12):
    part = frame.loc[frame["誤り種別"] == kind].copy()
    part = part.sort_values("閾値からの距離", ascending=False).head(n)
    rows = []
    for _, r in part.iterrows():
        rows.append({
            "企業ID": int(r["企業ID"]),
            "企業名": _clip_text(r["企業名"], 28),
            "業界": r["業界"],
            "従業員": int(r["従業員数"]),
            "p": f"{r['OOF確率']:.3f}",
            "Need": f"{r['Need_不満4']:.1f}",
            "Capacity%": f"{r['Capacity_ソフト投資売上比_pct']:.2f}",
            "Intent": (f"{r['llm_dx_execution_score']:.1f}"
                       if "llm_dx_execution_score" in r and
                       pd.notna(r["llm_dx_execution_score"]) else "—"),
            "DX展望（末尾）": _clip_tail_text(r["今後のDX展望"]),
        })
    return pd.DataFrame(rows)


def write_report(frame: pd.DataFrame, names: tuple[str, ...], y: np.ndarray,
                 blend: np.ndarray, threshold: float, alphas: list[float],
                 weights: np.ndarray, fingerprint: str,
                 report_path: Path):
    pred = (blend >= threshold).astype(int)
    score = {
        "auc": roc_auc_score(y, blend),
        "ap": average_precision_score(y, blend),
        "f1": f1_score(y, pred),
    }
    counts = frame["誤り種別"].value_counts()
    means = _mean_table(frame, names)
    industries = _industry_table(frame)
    fn = _case_table(frame, "FN")
    fp = _case_table(frame, "FP")
    weight_table = pd.DataFrame({"expert": names, "平均メタ重み": weights}) \
        .sort_values("平均メタ重み", ascending=False)

    text = f"""# 誤り分析 — discovery側のみ

生成条件: `seed={LOCKBOX_SEED}` / discovery {len(frame)}件 / lockbox {LOCKBOX_SIZE}件  
lockbox fingerprint: `{fingerprint}`

このレポートは固定分割の **discovery側だけ**で作成した。lockbox側の企業ID、ラベル、
予測値は表示・保存していない。ここから考案する特徴量は、定義を固定してからlockboxで
増分評価する。このレポート内の数値は探索用であり、最終的な汎化性能の推定値ではない。

## Discovery OOF

| AUC | AP | F1 | 閾値 | FN | FP | TP | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| {score['auc']:.4f} | {score['ap']:.4f} | {score['f1']:.4f} | {threshold:.3f} | {counts.get('FN', 0)} | {counts.get('FP', 0)} | {counts.get('TP', 0)} | {counts.get('TN', 0)} |

メタ層のfold別alpha: `{alphas}`

### 平均メタ重み

{_markdown_table(weight_table, float_digits=4)}

## 誤り種別ごとの平均

{_markdown_table(means, index=True, float_digits=3)}

数値差は仮説を作るための記述統計で、同じdiscovery上の有意差や特徴量採用根拠にはしない。

## 高確信FN

実際は購入だが、閾値より大きく下に予測した企業。

{_markdown_table(fn) if len(fn) else '該当なし'}

## 高確信FP

実際は非購入だが、閾値より大きく上に予測した企業。

{_markdown_table(fp) if len(fp) else '該当なし'}

## 業界別の誤り率

discovery内で8件以上ある業界のみ。探索用であり、業界補正を直接作る根拠にはしない。

{_markdown_table(industries, index=True, float_digits=3)}

## 読み方

- `FN/FP`の共通語をそのままフラグ化しない。
- まず「どの概念を取り違えたか」を文章で仮説化する。
- 特徴量定義・利用カラム・方向をlockboxを開ける前に固定する。
- 候補をdiscovery内CVで絞り、少数候補をまとめてlockboxで一度だけ比較する。
- 全誤分類の生データと8エキスパート予測は `{OUT_CSV}` にある。
"""
    report_path.write_text(text, encoding="utf-8")


def run(seed: int, lockbox_size: int, out_csv: Path,
        out_report: Path, out_protocol: Path):
    if seed != LOCKBOX_SEED or lockbox_size != LOCKBOX_SIZE:
        raise ValueError(
            "固定分割を不用意に変えないため、seed/sizeの変更は禁止。"
            "変更時はコード定数と実験記録を意図的に更新すること。"
        )
    train = pd.read_csv("data/train.csv")
    y_all = train["購入フラグ"].to_numpy(dtype=int)
    discovery, lockbox = fixed_split(y_all, seed, lockbox_size)
    fingerprint = lockbox_fingerprint(train.iloc[lockbox]["企業ID"].to_numpy())

    print(f"discovery={len(discovery)} / lockbox={len(lockbox)}")
    print(f"lockbox fingerprint={fingerprint}")
    print("lockboxのID・ラベル・予測は出力しません。")
    y, oof, blend, names, alphas, weights = discovery_oof(
        train, discovery, seed,
    )
    threshold = _best_threshold(y, blend)
    frame = _enrich(train, discovery, y, oof, blend, names, threshold)
    errors = frame.loc[frame["誤り種別"].isin(["FN", "FP"])].copy()
    errors = errors.sort_values(
        ["誤り種別", "閾値からの距離"], ascending=[True, False],
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_csv, index=False)
    write_report(frame, names, y, blend, threshold, alphas, weights,
                 fingerprint, out_report)
    protocol = {
        "version": 1,
        "purpose": "future_feature_increment_only",
        "seed": seed,
        "splitter": "sklearn.train_test_split(stratify=y)",
        "n_total": len(train),
        "n_discovery": len(discovery),
        "n_lockbox": len(lockbox),
        "lockbox_id_sha256": fingerprint,
        "model": "exp032-like: concatemb + E7 LR + E0b LR(C=0.03)",
        "policy": "Do not inspect or export lockbox IDs, labels, predictions, or row-level errors before preregistration.",
        "outputs": [str(out_csv), str(out_report)],
    }
    out_protocol.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"保存: {out_csv} ({len(errors)}誤分類)")
    print(f"保存: {out_report}")
    print(f"保存: {out_protocol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=LOCKBOX_SEED)
    parser.add_argument("--lockbox-size", type=int, default=LOCKBOX_SIZE)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    parser.add_argument("--out-protocol", type=Path, default=OUT_PROTOCOL)
    args = parser.parse_args()
    run(args.seed, args.lockbox_size, args.out_csv,
        args.out_report, args.out_protocol)
