"""現行H2-both4モデルの誤りを、E2の反実仮想とexpert合意から分解する。"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import _scores, blend_oof  # noqa: E402
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from expert_groups import SURVEY_COLS  # noqa: E402


SEED = 42
CACHE = "exp/_experts_seed42_concatemb_crosslr_lin0.03_llmcore4e0onlyh2both4.npz"
OUT_CSV = "exp/current_best_error_groups.csv"
OUT_MD = "documents/current_best_error_analysis.md"
E2 = "E2_survey"


def fmt(v):
    if pd.isna(v):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}"
    return str(v).replace("|", "／").replace("\n", " ")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    raw = pd.read_csv("data/train.csv")
    llm = pd.read_csv("data/llm_make_or_buy/train_make_or_buy_features.csv")
    data = np.load(CACHE)
    y = data["y"].astype(int)
    assert np.array_equal(y, raw["購入フラグ"].to_numpy())
    oof = {n: data[f"oof_{n}"] for n in BASE_EXPERTS}
    te = {n: data[f"te_{n}"] for n in BASE_EXPERTS}

    full, _, weights, alphas = blend_oof(y, oof, te, SEED, experts=BASE_EXPERTS)
    without = tuple(n for n in BASE_EXPERTS if n != E2)
    no_e2, _, _, _ = blend_oof(y, oof, te, SEED, experts=without)
    full_score, no_e2_score = _scores(y, full), _scores(y, no_e2)
    th, th_no = full_score["th"], no_e2_score["th"]
    pred, pred_no = full >= th, no_e2 >= th_no

    expert_th = {n: _scores(y, oof[n])["th"] for n in BASE_EXPERTS}
    votes = np.column_stack([oof[n] >= expert_th[n] for n in BASE_EXPERTS])
    all_miss = (y == 1) & (votes.sum(axis=1) == 0)
    rescued = (y == 1) & pred & ~pred_no
    harmed = (y == 0) & ~pred_no & pred
    fn = (y == 1) & ~pred
    fp = (y == 0) & pred
    margin = full - th
    high_fn = fn & (margin <= np.quantile(margin[fn], .25) if fn.any() else False)
    high_fp = fp & (margin >= np.quantile(margin[fp], .75) if fp.any() else False)

    labels = np.full(len(y), "correct_other", dtype=object)
    labels[fn] = "FN"
    labels[fp] = "FP"
    labels[high_fn] = "high_conf_FN"
    labels[high_fp] = "high_conf_FP"
    labels[rescued] = "E2_rescued_TP"
    labels[harmed] = "E2_harmed_FP"
    labels[all_miss] = "all_experts_missed_positive"

    keep_raw = ["企業ID", "企業名", "業界", "従業員数", "事業所数", "工場数", "店舗数",
                "売上", "営業利益", "営業CF"] + list(SURVEY_COLS)
    out = raw[keep_raw].copy().merge(llm, on="企業ID", how="left")
    out.insert(2, "error_group", labels)
    out["y"] = y
    out["p_full"] = full
    out["p_without_E2"] = no_e2
    out["E2_delta"] = full - no_e2
    out["expert_positive_votes"] = votes.sum(axis=1)
    for n in BASE_EXPERTS:
        out[f"p_{n}"] = oof[n]
    focus = labels != "correct_other"
    out.loc[focus].sort_values(["error_group", "p_full"]).to_csv(OUT_CSV, index=False)

    group_rows = []
    for name in ["E2_rescued_TP", "E2_harmed_FP", "all_experts_missed_positive",
                 "high_conf_FN", "high_conf_FP", "FN", "FP"]:
        m = labels == name
        group_rows.append({"群": name, "件数": int(m.sum()),
                           "平均p": full[m].mean() if m.any() else np.nan,
                           "E2除外p": no_e2[m].mean() if m.any() else np.nan,
                           "E2差分": (full[m] - no_e2[m]).mean() if m.any() else np.nan,
                           "平均賛成expert数": votes[m].sum(axis=1).mean() if m.any() else np.nan})
    groups = pd.DataFrame(group_rows)

    # 群と同じラベルの母集団との差を、解釈しやすい既存・LLM数値だけで順位付けする。
    numeric = list(SURVEY_COLS) + [
        "従業員数", "事業所数", "工場数", "店舗数", "売上", "営業利益", "営業CF",
        "need_score", "make_limitation_score", "buy_openness_score",
        "internal_training_capability_score", "implementation_stage_ordinal",
        "evidence_strength", "make_or_buy_gap", "internal_training_gap"]
    contrasts = []
    for name in ["E2_rescued_TP", "E2_harmed_FP", "all_experts_missed_positive",
                 "high_conf_FN", "high_conf_FP"]:
        m = labels == name
        if not m.any():
            continue
        reference = y == (1 if name in {"E2_rescued_TP", "all_experts_missed_positive", "high_conf_FN"} else 0)
        for col in numeric:
            a = pd.to_numeric(out.loc[m, col], errors="coerce")
            b = pd.to_numeric(out.loc[reference, col], errors="coerce")
            sd = b.std()
            effect = (a.mean() - b.mean()) / sd if pd.notna(sd) and sd > 0 else np.nan
            contrasts.append({"群": name, "特徴": col, "群平均": a.mean(),
                              "同ラベル平均": b.mean(), "標準化差": effect})
    contrast = pd.DataFrame(contrasts)
    top = (contrast.assign(abs差=contrast["標準化差"].abs())
           .sort_values(["群", "abs差"], ascending=[True, False])
           .groupby("群", sort=False).head(6).drop(columns="abs差"))

    score_df = pd.DataFrame([
        {"構成": "現行H2-both4", **full_score},
        {"構成": "E2除外・再学習", **no_e2_score},
    ])
    weight_df = pd.DataFrame({"expert": BASE_EXPERTS, "weight": weights}).sort_values("weight", ascending=False)
    lines = [
        "# 現行ベスト H2-both4 誤り分解",
        "",
        "seed 42のOOFのみを使った診断。E2救済・毀損は、E2を除外してメタモデルを同じfoldで再学習した構成との比較で定義する。因果効果の証明ではなく、次の仮説を作るための反実仮想診断である。",
        "",
        "## 基準性能", "", md_table(score_df), "",
        f"現行閾値 `{th:.3f}`、E2除外閾値 `{th_no:.3f}`。メタalpha: `{alphas}`。", "",
        "## メタ重み", "", md_table(weight_df), "",
        "## 誤り群", "", md_table(groups), "",
        "注意: `all_experts_missed_positive` は各expertをそれぞれのOOF最適閾値で二値化し、8本すべてが負とした正例。群は表示上排他的に上書きされるため、FN/FP行は高確信群などを除いた残り。",
        "", "## 群を特徴づける差（同じ正解ラベル平均との差）", "", md_table(top), "",
        f"全対象行とexpert予測は `{OUT_CSV}` に保存。標準化差は探索用であり、特徴量採用の検証結果ではない。",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("exp/current_best_error_summary.json", "w", encoding="utf-8") as f:
        json.dump({"full": full_score, "without_e2": no_e2_score,
                   "counts": {r["群"]: r["件数"] for r in group_rows}}, f,
                  ensure_ascii=False, indent=2)
    print(score_df.to_string(index=False))
    print(groups.to_string(index=False))
    print(f"wrote {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()
