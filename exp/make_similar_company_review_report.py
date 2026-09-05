"""現行1SEモデルに直結した「似ているのに購入結果が逆」の企業ペアを作る。

3つの特徴空間から10組ずつ、企業重複なしで選ぶ。

* 表形式: E0アンカーの数値68列を中央値補完・標準化した cosine 類似度
* 組織・概要: 組織図+企業概要の OpenAI embedding cosine 類似度
* DX展望: 今後のDX展望の OpenAI embedding cosine 類似度

ペア候補は現行 seed=42 / 1SE メタ合成の OOF が 0.1〜0.9 にあり、
OOF確率差が0.05以下、かつ購入フラグが逆のものに限定する。選定は確率帯を
低・中・高=3・4・3組/特徴空間に固定した混合整数計画で、各帯内の類似順位を
最大化する。レビュー資料にはラベルとOOF確率を出さず、回答キーへ分離する。

Run:
  .venv/bin/python exp/make_similar_company_review_report.py \
    --output-dir outputs/<thread-id>
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import _inputs  # noqa: E402
from diagnose_band_local_structure import _blend  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from embedding_features import (  # noqa: E402
    CONCAT_SLUGS, load_concat_embeddings, load_embeddings,
)
from ensemble_experts import THS  # noqa: E402


LENS_ORDER = ("tabular", "org_overview", "dx")
LENS_META = {
    "tabular": {
        "code": "TAB",
        "name": "表形式（現行アンカー数値）",
        "definition": "E0アンカーの数値68列を中央値補完・標準化したcosine類似度",
    },
    "org_overview": {
        "code": "ORG",
        "name": "組織図＋企業概要",
        "definition": "組織図と企業概要を連結したOpenAI embeddingのcosine類似度",
    },
    "dx": {
        "code": "DX",
        "name": "今後のDX展望",
        "definition": "今後のDX展望のOpenAI embeddingのcosine類似度",
    },
}
BUCKET_ORDER = ("low", "mid", "high")
BUCKET_META = {
    "low": {"name": "低確率側", "quota": 3},
    "mid": {"name": "中間", "quota": 4},
    "high": {"name": "高確率側", "quota": 3},
}

RAW_BASIC = (
    "企業ID", "企業名", "業界", "上場種別", "特徴", "従業員数",
    "事業所数", "工場数", "店舗数",
)
RAW_FINANCE = (
    "資本金", "総資産", "流動資産", "固定資産", "負債", "短期借入金",
    "長期借入金", "純資産", "自己資本", "売上", "営業利益", "経常利益",
    "当期純利益", "営業CF", "減価償却費", "運転資本変動", "投資CF",
    "有形固定資産変動", "無形固定資産変動(ソフトウェア関連)",
)
SURVEYS = tuple(f"アンケート{digit}" for digit in "１２３４５６７８９") + (
    "アンケート１０", "アンケート１１",
)
RATIOS = ("営業利益率", "経常利益率", "純利益率", "ROA", "FCF比率", "現金ROA", "自己資本比率")
LLM_AXES = (
    "LLM_need_score", "LLM_make_limitation_score",
    "LLM_buy_openness_score", "LLM_internal_training_capability_score",
)


def _bucket(value: float) -> str:
    if value < 0.3:
        return "low"
    if value < 0.7:
        return "mid"
    return "high"


def _native(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _fmt(value, field: str | None = None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if field in RATIOS:
        return f"{float(value):.1%}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.3f}"
    return str(value)


def _table_text(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _candidate_pairs(y, p, band_index, similarity, max_probability_gap):
    rows = []
    for left in range(len(band_index)):
        i = int(band_index[left])
        for right in range(left + 1, len(band_index)):
            j = int(band_index[right])
            if y[i] == y[j] or abs(float(p[i] - p[j])) > max_probability_gap:
                continue
            mean_p = float((p[i] + p[j]) / 2)
            rows.append({
                "i": i,
                "j": j,
                "similarity": float(similarity[left, right]),
                "p_mean": mean_p,
                "p_gap": abs(float(p[i] - p[j])),
                "bucket": _bucket(mean_p),
            })
    return rows


def _select_disjoint(candidates_by_lens, company_count):
    variables = []
    for lens in LENS_ORDER:
        for bucket in BUCKET_ORDER:
            group = [c for c in candidates_by_lens[lens] if c["bucket"] == bucket]
            group.sort(key=lambda row: row["similarity"], reverse=True)
            denominator = max(1, len(group) - 1)
            for rank, candidate in enumerate(group):
                variables.append({
                    **candidate,
                    "lens": lens,
                    "rank_in_group": rank + 1,
                    "group_size": len(group),
                    "quality": 1.0 - rank / denominator,
                })

    group_rows = {(lens, bucket): n for n, (lens, bucket) in enumerate(
        (lens, bucket) for lens in LENS_ORDER for bucket in BUCKET_ORDER
    )}
    company_offset = len(group_rows)
    matrix_rows, matrix_cols, matrix_values = [], [], []
    for col, candidate in enumerate(variables):
        matrix_rows.append(group_rows[(candidate["lens"], candidate["bucket"])])
        matrix_cols.append(col)
        matrix_values.append(1.0)
        for company in (candidate["i"], candidate["j"]):
            matrix_rows.append(company_offset + company)
            matrix_cols.append(col)
            matrix_values.append(1.0)

    constraint_count = company_offset + company_count
    matrix = coo_array(
        (matrix_values, (matrix_rows, matrix_cols)),
        shape=(constraint_count, len(variables)),
    ).tocsc()
    lower = np.zeros(constraint_count)
    upper = np.ones(constraint_count)
    for (lens, bucket), row in group_rows.items():
        lower[row] = BUCKET_META[bucket]["quota"]
        upper[row] = BUCKET_META[bucket]["quota"]

    # 各特徴空間・確率帯の順位を同じ0〜1尺度で最大化し、空間間の尺度差を消す。
    objective = -np.asarray([
        row["quality"] + 1e-6 * row["similarity"] for row in variables
    ])
    result = milp(
        c=objective,
        integrality=np.ones(len(variables)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 30.0},
    )
    if not result.success:
        raise RuntimeError(f"ペア選定MILPが解けませんでした: {result.message}")
    selected = [row for row, value in zip(variables, result.x) if value > 0.5]
    if len(selected) != 30:
        raise AssertionError(f"選定数が30ではありません: {len(selected)}")
    return selected


def _company_record(index, train, engineered, anchor_numeric):
    raw = train.iloc[index]
    xrow = engineered.iloc[index]
    return {
        "row_index": int(index),
        "basic": {field: _native(raw[field]) for field in RAW_BASIC},
        "finance": {field: _native(raw[field]) for field in RAW_FINANCE},
        "survey": {field: _native(raw[field]) for field in SURVEYS},
        "ratios": {field: _native(xrow[field]) for field in RATIOS},
        "llm_axes": {field: _native(xrow[field]) for field in LLM_AXES},
        "anchor_numeric": {field: _native(xrow[field]) for field in anchor_numeric},
        "texts": {
            "企業概要": str(raw["企業概要"]) if not pd.isna(raw["企業概要"]) else "",
            "組織図": str(raw["組織図"]) if not pd.isna(raw["組織図"]) else "",
            "今後のDX展望": str(raw["今後のDX展望"]) if not pd.isna(raw["今後のDX展望"]) else "",
        },
    }


def _orient_and_label(selected, train, engineered, anchor_numeric, y, p, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for lens in LENS_ORDER:
        lens_rows = [row for row in selected if row["lens"] == lens]
        lens_rows.sort(key=lambda row: (
            BUCKET_ORDER.index(row["bucket"]), -row["similarity"]
        ))
        positive_on_a = set(rng.choice(len(lens_rows), size=5, replace=False).tolist())
        for order, row in enumerate(lens_rows, start=1):
            positive = row["i"] if y[row["i"]] == 1 else row["j"]
            negative = row["j"] if positive == row["i"] else row["i"]
            if order - 1 in positive_on_a:
                a_index, b_index = positive, negative
            else:
                a_index, b_index = negative, positive
            pairs.append({
                "pair_id": f"{LENS_META[lens]['code']}-{order:02d}",
                "lens": lens,
                "lens_name": LENS_META[lens]["name"],
                "similarity": row["similarity"],
                "similarity_percentile": 1.0 - ((row["rank_in_group"] - 1) / max(1, row["group_size"])),
                "rank_in_group": row["rank_in_group"],
                "group_size": row["group_size"],
                "bucket": row["bucket"],
                "a": _company_record(a_index, train, engineered, anchor_numeric),
                "b": _company_record(b_index, train, engineered, anchor_numeric),
                "answer": {
                    "a_label": int(y[a_index]),
                    "b_label": int(y[b_index]),
                    "a_probability": float(p[a_index]),
                    "b_probability": float(p[b_index]),
                    "probability_gap": abs(float(p[a_index] - p[b_index])),
                    "probability_mean": float((p[a_index] + p[b_index]) / 2),
                },
            })
    return pairs


def _write_text_comparison(handle, title, a_name, a_text, b_name, b_text, opened):
    flag = " open" if opened else ""
    handle.write(f"<details{flag}>\n<summary>{html.escape(title)}</summary>\n\n")
    handle.write("<table><tr>")
    handle.write(f"<th style=\"width:50%\">A: {html.escape(a_name)}</th>")
    handle.write(f"<th style=\"width:50%\">B: {html.escape(b_name)}</th></tr><tr>")
    handle.write("<td><div style=\"white-space:pre-wrap\">")
    handle.write(html.escape(a_text))
    handle.write("</div></td><td><div style=\"white-space:pre-wrap\">")
    handle.write(html.escape(b_text))
    handle.write("</div></td></tr></table>\n\n</details>\n\n")


def _write_review_report(
        path, pairs, anchor_numeric, candidate_counts, show_labels=False):
    with path.open("w", encoding="utf-8") as handle:
        title = "ラベル付きレビュー" if show_labels else "ブラインドレビュー"
        handle.write(f"# 似た企業ペア・{title}\n\n")
        handle.write("> **コンペ配布データを含む非公開資料です。開催中は外部公開・再配布しないでください。**\n\n")
        if show_labels:
            handle.write("A/Bそれぞれの購入フラグを一覧と各ペア詳細に表示した施策検討版です。**購入（1）** と **未購入（0）** の差を直接見比べられます。30組・60社は重複なしです。\n\n")
        else:
            handle.write("購入フラグとOOF確率を完全に除いたレビュー版です。30組・60社は重複なしで、各組は実購入結果が逆ですが、どちらが購入かはここには表示していません。\n\n")
        handle.write("## 見方\n\n")
        if show_labels:
            handle.write("1. 一覧の緑色相当の **購入（1）** と **未購入（0）** を確認します。\n")
            handle.write("2. ペア詳細では、選定レンズを先に見てから、数値・文章で『逆の結果を説明する差』を探します。\n")
            handle.write("3. 差を、既存列・生成痕跡・交互作用・未観測変数のどれで表現できるかメモします。\n\n")
        else:
            handle.write("1. まず一覧で各組について `A / B / 判別不能` を決め、確信度1〜5を記録します。\n")
            handle.write("2. ペア詳細では、選定レンズを先に見てから、他の数値・文章で『逆の結果を説明する差』を探します。\n")
            handle.write("3. 決め手になった差を、既存列・生成痕跡・未観測変数のどれかとしてメモします。全30組が終わるまで回答キーを開かないでください。\n\n")
        handle.write("## 現行処理における類似度\n\n")
        for lens in LENS_ORDER:
            handle.write(f"- **{LENS_META[lens]['name']}**: {LENS_META[lens]['definition']}\n")
        handle.write("\n候補条件は seed=42 の現行8エキスパート＋1SEメタ合成のOOFが0.1〜0.9、ペア内確率差≤0.05、実購入結果が逆です。低・中・高確率帯から各レンズ3・4・3組を選びました。\n\n")
        handle.write("## レビュー一覧\n\n")
        if show_labels:
            handle.write("| Pair | 類似レンズ | 類似度 | A企業 | **A結果** | A業界 | B企業 | **B結果** | B業界 | 着目差 / 新特徴量仮説 |\n")
            handle.write("|---|---|---:|---|:---:|---|---|:---:|---|---|\n")
        else:
            handle.write("| Pair | 類似レンズ | 類似度 | 上位% | A企業 | A業界 | B企業 | B業界 | 同業界 | 人の判定 | 確信度 | 決め手 / 新特徴量仮説 |\n")
            handle.write("|---|---|---:|---:|---|---|---|---|:---:|---|---:|---|\n")
        for pair in pairs:
            a, b = pair["a"]["basic"], pair["b"]["basic"]
            if show_labels:
                a_result = "🟢 **購入（1）**" if pair["answer"]["a_label"] else "⚪ 未購入（0）"
                b_result = "🟢 **購入（1）**" if pair["answer"]["b_label"] else "⚪ 未購入（0）"
                handle.write(
                    f"| {pair['pair_id']} | {_table_text(pair['lens_name'])} | {pair['similarity']:.3f} | "
                    f"{_table_text(a['企業名'])} | {a_result} | {_table_text(a['業界'])} | "
                    f"{_table_text(b['企業名'])} | {b_result} | {_table_text(b['業界'])} |  |\n"
                )
            else:
                same = "○" if a["業界"] == b["業界"] else "—"
                handle.write(
                    f"| {pair['pair_id']} | {_table_text(pair['lens_name'])} | {pair['similarity']:.3f} | "
                    f"{pair['similarity_percentile']:.1%} | {_table_text(a['企業名'])} | {_table_text(a['業界'])} | "
                    f"{_table_text(b['企業名'])} | {_table_text(b['業界'])} | {same} |  |  |  |\n"
                )

        handle.write("\n## ペア詳細\n\n")
        for pair in pairs:
            a, b = pair["a"], pair["b"]
            an, bn = a["basic"]["企業名"], b["basic"]["企業名"]
            handle.write(f"## {pair['pair_id']} — {pair['lens_name']}（類似度 {pair['similarity']:.3f}）\n\n")
            if show_labels:
                a_result = "🟢 購入（1）" if pair["answer"]["a_label"] else "⚪ 未購入（0）"
                b_result = "🟢 購入（1）" if pair["answer"]["b_label"] else "⚪ 未購入（0）"
                handle.write(f"> **結果: A={a_result} / B={b_result}**\n\n")
            handle.write("### 基本属性\n\n| 項目 | A | B |\n|---|---:|---:|\n")
            if show_labels:
                handle.write(
                    f"| **購入フラグ** | **{pair['answer']['a_label']}** | **{pair['answer']['b_label']}** |\n"
                )
            for field in RAW_BASIC:
                handle.write(f"| {_table_text(field)} | {_table_text(_fmt(a['basic'][field], field))} | {_table_text(_fmt(b['basic'][field], field))} |\n")

            numeric_open = pair["lens"] == "tabular"
            handle.write(f"\n<details{' open' if numeric_open else ''}>\n<summary>現行アンカー数値 {len(anchor_numeric)}項目</summary>\n\n")
            handle.write("| 項目 | A | B |\n|---|---:|---:|\n")
            for field in anchor_numeric:
                handle.write(f"| {_table_text(field)} | {_fmt(a['anchor_numeric'][field], field)} | {_fmt(b['anchor_numeric'][field], field)} |\n")
            handle.write("\n</details>\n\n")

            _write_text_comparison(
                handle, "企業概要", an, a["texts"]["企業概要"], bn,
                b["texts"]["企業概要"], pair["lens"] == "org_overview",
            )
            _write_text_comparison(
                handle, "組織図", an, a["texts"]["組織図"], bn,
                b["texts"]["組織図"], pair["lens"] == "org_overview",
            )
            _write_text_comparison(
                handle, "今後のDX展望", an, a["texts"]["今後のDX展望"], bn,
                b["texts"]["今後のDX展望"], pair["lens"] == "dx",
            )
            if show_labels:
                handle.write("**購入／未購入を分けたと考える差:** ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿  \n")
                handle.write("**新しい特徴量・交互作用・未観測変数の仮説:** ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿\n\n")
            else:
                handle.write("**人の判定:** ＿＿＿　 **確信度:** ＿＿ / 5  \n")
                handle.write("**決め手:** ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿  \n")
                handle.write("**新しい特徴量・未観測変数の仮説:** ＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿\n\n")
            handle.write("---\n\n")

        handle.write("## 選定QC\n\n")
        handle.write(f"- 元の学習データ: 742社\n- 選定: 30組 / 60社（企業重複なし）\n")
        for lens in LENS_ORDER:
            count = sum(pair["lens"] == lens for pair in pairs)
            handle.write(f"- {LENS_META[lens]['name']}: {count}組（候補 {candidate_counts[lens]:,}組）\n")
        handle.write(f"- 作成日: {date.today().isoformat()}\n")


def _write_answer_key(path, pairs, threshold):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 似た企業ペア・回答キー\n\n")
        handle.write("> **先にブラインドレビューを完了してください。コンペ配布データを含むため外部公開・再配布禁止です。**\n\n")
        handle.write(f"現行1SE OOFの全体F1最適閾値は `{threshold:.3f}` です。OOF確率は各行が学習時に見られていないfold予測ですが、同じ学習データ内の診断値であり、未知データ性能の保証ではありません。\n\n")
        handle.write("| Pair | 類似レンズ | A企業 | A実績 | A OOF | B企業 | B実績 | B OOF | Δp | 正解 | モデル判定(A/B) |\n")
        handle.write("|---|---|---|---:|---:|---|---:|---:|---:|:---:|:---:|\n")
        for pair in pairs:
            a = pair["a"]["basic"]
            b = pair["b"]["basic"]
            answer = pair["answer"]
            correct = "A" if answer["a_label"] == 1 else "B"
            model_a = int(answer["a_probability"] >= threshold)
            model_b = int(answer["b_probability"] >= threshold)
            handle.write(
                f"| {pair['pair_id']} | {_table_text(pair['lens_name'])} | {_table_text(a['企業名'])} | "
                f"{answer['a_label']} | {answer['a_probability']:.3f} | {_table_text(b['企業名'])} | "
                f"{answer['b_label']} | {answer['b_probability']:.3f} | {answer['probability_gap']:.3f} | "
                f"**{correct}** | {model_a}/{model_b} |\n"
            )
        handle.write("\n`実績 1=購入、0=未購入`。モデル判定が両社同じになっている組ほど、現在の特徴量・合成では分けられていない典型です。\n")


def _write_json(path, pairs, threshold, seed, max_probability_gap, anchor_numeric):
    blind_pairs = []
    answers = []
    for pair in pairs:
        blind = {key: value for key, value in pair.items() if key != "answer"}
        blind_pairs.append(blind)
        answers.append({"pair_id": pair["pair_id"], **pair["answer"]})
    payload = {
        "selection": {
            "config": "h2h7both4",
            "meta_rule": "1se",
            "seed": seed,
            "band": [0.1, 0.9],
            "max_probability_gap": max_probability_gap,
            "f1_threshold": threshold,
            "anchor_numeric_columns": anchor_numeric,
        },
        "blind_pairs": blind_pairs,
        "answer_key": answers,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(output_dir: str, seed: int, orientation_seed: int, max_probability_gap: float):
    train = pd.read_csv("data/train.csv")
    X, y_ref, _, _, _, columns = _inputs()
    y, expert_logits, _ = _load(CONFIGS["h2h7both4"], seed)
    if not np.array_equal(y, y_ref):
        raise AssertionError("train.csvと専門家キャッシュのラベル順が一致しません")
    probability = _blend(y, expert_logits, seed)
    threshold = float(THS[int(np.argmax([
        f1_score(y, (probability >= value).astype(int)) for value in THS
    ]))])

    anchor_numeric = list(X[columns[0]].select_dtypes(include=[np.number]).columns)
    tabular = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
    ).fit_transform(X[anchor_numeric])
    org_overview, _ = load_concat_embeddings(CONCAT_SLUGS)
    dx, _ = load_embeddings("dx_outlook")
    spaces = {"tabular": tabular, "org_overview": org_overview, "dx": dx}

    band_mask = (probability >= 0.1) & (probability <= 0.9)
    band_index = np.flatnonzero(band_mask)
    candidates_by_lens = {}
    for lens, features in spaces.items():
        similarity = cosine_similarity(features[band_index])
        candidates_by_lens[lens] = _candidate_pairs(
            y, probability, band_index, similarity, max_probability_gap,
        )

    selected = _select_disjoint(candidates_by_lens, len(train))
    pairs = _orient_and_label(
        selected, train, X, anchor_numeric, y, probability, orientation_seed,
    )

    # 不変条件: 30組・60社、各空間10、各帯3/4/3、逆ラベル、A購入5/空間。
    indices = [pair[side]["row_index"] for pair in pairs for side in ("a", "b")]
    assert len(pairs) == 30 and len(set(indices)) == 60
    assert Counter(pair["lens"] for pair in pairs) == Counter({lens: 10 for lens in LENS_ORDER})
    for lens in LENS_ORDER:
        lens_pairs = [pair for pair in pairs if pair["lens"] == lens]
        assert Counter(pair["bucket"] for pair in lens_pairs) == Counter({"low": 3, "mid": 4, "high": 3})
        assert sum(pair["answer"]["a_label"] for pair in lens_pairs) == 5
    assert all(pair["answer"]["a_label"] != pair["answer"]["b_label"] for pair in pairs)
    assert all(pair["answer"]["probability_gap"] <= max_probability_gap + 1e-12 for pair in pairs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    blind_path = out / "similar_company_review_blind.md"
    labeled_path = out / "similar_company_review_with_labels.md"
    answer_path = out / "similar_company_review_answer_key.md"
    json_path = out / "similar_company_review_data.json"
    candidate_counts = {lens: len(rows) for lens, rows in candidates_by_lens.items()}
    _write_review_report(
        blind_path, pairs, anchor_numeric, candidate_counts, show_labels=False,
    )
    _write_review_report(
        labeled_path, pairs, anchor_numeric, candidate_counts, show_labels=True,
    )
    _write_answer_key(answer_path, pairs, threshold)
    _write_json(json_path, pairs, threshold, seed, max_probability_gap, anchor_numeric)

    print(f"band rows: {int(band_mask.sum())}")
    print(f"threshold: {threshold:.3f}")
    print("candidate counts:", candidate_counts)
    print("selected:")
    for pair in pairs:
        a = pair["a"]["basic"]
        b = pair["b"]["basic"]
        print(
            f"{pair['pair_id']} {pair['similarity']:.4f} "
            f"A={a['企業ID']}:{a['企業名']} B={b['企業ID']}:{b['企業名']} "
            f"key={pair['answer']['a_label']}/{pair['answer']['b_label']} "
            f"p={pair['answer']['a_probability']:.3f}/{pair['answer']['b_probability']:.3f}"
        )
    print(f"saved: {blind_path}")
    print(f"saved: {labeled_path}")
    print(f"saved: {answer_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orientation-seed", type=int, default=20260830)
    parser.add_argument("--max-probability-gap", type=float, default=0.05)
    args = parser.parse_args()
    main(args.output_dir, args.seed, args.orientation_seed, args.max_probability_gap)
