"""H12: split `今後のDX展望` into current/future text fields.

The hypothesis is that the document has a contrastive structure:
current state versus future plan.  This script compares a single full-text
TF-IDF space with separate current/future TF-IDF spaces.

Usage:
  python3 exp/compare_h12_tense.py [n_seeds]
"""
import datetime
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "購入フラグ"
TEXT_COL = "今後のDX展望"
N_SPLITS = 5
N_JOBS = 8
THS = np.arange(0.05, 0.95, 0.005)

# random_state 必須。理由は text_features.LR_PARAMS のコメントを参照。
LR_PARAMS = dict(C=1.0, solver="liblinear", dual=True, max_iter=3000,
                 random_state=0)
BASE_TFIDF_PARAMS = dict(min_df=5, sublinear_tf=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "exp")
LOG = os.path.join(EXP_DIR, "hypothesis_log.md")

_NUM = re.compile(r"[0-9]+(?:[,.][0-9]+)*")
_SPACE = re.compile(r"[\s　]+")
_SENT = re.compile(r"(?<=。)")

CURRENT_MARK = (
    "これまで", "従来", "現状", "既存", "とどま", "に過ぎ", "でした",
    "ました", "きました", "残っており", "が実情", "限定的", "不十分",
)
FUTURE_MARK = (
    "今後", "まいります", "計画", "方針", "予定", "検討", "していく",
    "図る", "目指", "する考え", "拡大し", "整備し", "強化", "推進",
)


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = _NUM.sub("0", s)
    s = _SPACE.sub(" ", s)
    return s.strip()


def split_sentences(text):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def tense_of(sent):
    current = any(m in sent for m in CURRENT_MARK)
    future = any(m in sent for m in FUTURE_MARK)
    if future and not current:
        return "future"
    if current and not future:
        return "current"
    return "both"


def tense_parts(text):
    current, future = [], []
    for sent in split_sentences(text):
        tense = tense_of(sent)
        if tense in ("current", "both"):
            current.append(sent)
        if tense in ("future", "both"):
            future.append(sent)
    return " ".join(current), " ".join(future)


def best_f1(y, p):
    f1s = np.array([f1_score(y, p >= t) for t in THS])
    i = int(np.argmax(f1s))
    return float(f1s[i]), float(THS[i])


def vectorize_fields(fields, tr, va, analyzer, ngram_range):
    Xt, Xv = [], []
    params = dict(BASE_TFIDF_PARAMS)
    params.update(analyzer=analyzer, ngram_range=ngram_range)
    for field in fields:
        v = TfidfVectorizer(**params)
        Xt.append(v.fit_transform(field[tr]))
        Xv.append(v.transform(field[va]))
    if len(Xt) == 1:
        return Xt[0], Xv[0]
    return hstack(Xt).tocsr(), hstack(Xv).tocsr()


def _one_fold(candidate, y, seed, fold, tr, va):
    Xtr, Xva = vectorize_fields(
        candidate["fields"], tr, va, candidate["analyzer"],
        candidate["ngram_range"]
    )
    model = LogisticRegression(**LR_PARAMS).fit(Xtr, y[tr])
    pred = model.predict_proba(Xva)[:, 1]
    fold_f1, fold_th = best_f1(y[va], pred)
    return seed, fold, va, pred, fold_f1, fold_th


def evaluate_candidate(candidate, y, seeds):
    jobs = []
    for seed in seeds:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            jobs.append((seed, fold, tr, va))

    out = Parallel(n_jobs=N_JOBS)(
        delayed(_one_fold)(candidate, y, seed, fold, tr, va)
        for seed, fold, tr, va in jobs
    )

    rows = []
    fold_rows = []
    for seed in seeds:
        oof = np.zeros(len(y))
        seed_folds = []
        for s, fold, va, pred, fold_f1, fold_th in out:
            if s != seed:
                continue
            oof[va] = pred
            seed_folds.append((fold, fold_f1, fold_th))

        f1, th = best_f1(y, oof)
        rows.append(dict(
            name=candidate["name"],
            setting=candidate["setting"],
            seed=seed,
            auc=roc_auc_score(y, oof),
            ap=average_precision_score(y, oof),
            f1=f1,
            th=th,
            fold_f1=",".join(f"{v:.4f}" for _, v, _ in sorted(seed_folds)),
            fold_th=",".join(f"{v:.3f}" for _, _, v in sorted(seed_folds)),
        ))
        for fold, fold_f1, fold_th in seed_folds:
            fold_rows.append(dict(
                name=candidate["name"],
                setting=candidate["setting"],
                seed=seed,
                fold=fold,
                fold_f1=fold_f1,
                fold_th=fold_th,
            ))

    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def summarize(df):
    return df.groupby(["setting", "name"], sort=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        th_mean=("th", "mean"),
        th_std=("th", "std"),
    ).reset_index()


def format_table(summary):
    lines = [
        "setting       name                   AUC             AP              F1              th",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"{r.setting:12s}  {r.name:21s}  "
            f"{r.auc_mean:.4f}±{r.auc_std:.4f}  "
            f"{r.ap_mean:.4f}±{r.ap_std:.4f}  "
            f"{r.f1_mean:.4f}±{r.f1_std:.4f}  "
            f"{r.th_mean:.3f}±{r.th_std:.3f}"
        )
    return "\n".join(lines)


def make_candidates(full, current, future):
    field_sets = [
        ("H12-1 full", [full]),
        ("H12-2 current", [current]),
        ("H12-3 future", [future]),
        ("H12-4 current|future", [current, future]),
        ("H12-5 full|future", [full, future]),
        ("H12-6 full|current|future", [full, current, future]),
    ]
    settings = [
        ("char_wb(1,3)", "char_wb", (1, 3)),
        ("char(2,6)", "char", (2, 6)),
    ]
    out = []
    for setting, analyzer, ngram_range in settings:
        for name, fields in field_sets:
            out.append(dict(
                setting=setting,
                analyzer=analyzer,
                ngram_range=ngram_range,
                name=name,
                fields=fields,
            ))
    return out


def main(n_seeds=5):
    train = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    y = train[TARGET].values
    full = np.array([normalize(x) for x in train[TEXT_COL].fillna("")],
                    dtype=object)
    parts = [tense_parts(x) for x in full]
    current = np.array([x[0] for x in parts], dtype=object)
    future = np.array([x[1] for x in parts], dtype=object)

    print(
        "length mean: "
        f"full={np.mean([len(x) for x in full]):.0f}, "
        f"current={np.mean([len(x) for x in current]):.0f}, "
        f"future={np.mean([len(x) for x in future]):.0f}"
    )
    print(
        "empty rows: "
        f"current={int(np.sum(current == ''))}/{len(current)}, "
        f"future={int(np.sum(future == ''))}/{len(future)}"
    )

    seeds = list(range(n_seeds))
    rows, fold_rows = [], []
    for candidate in make_candidates(full, current, future):
        print(f"--- {candidate['setting']} {candidate['name']} ---",
              flush=True)
        r, f = evaluate_candidate(candidate, y, seeds)
        rows.append(r)
        fold_rows.append(f)

    df = pd.concat(rows, ignore_index=True)
    fold_df = pd.concat(fold_rows, ignore_index=True)
    summary = summarize(df)

    df.to_csv(os.path.join(EXP_DIR, "_h12_tense_scores.csv"), index=False)
    fold_df.to_csv(os.path.join(EXP_DIR, "_h12_tense_fold_scores.csv"),
                   index=False)
    summary.to_csv(os.path.join(EXP_DIR, "_h12_tense_summary.csv"),
                   index=False)

    table = format_table(summary)
    out = f"H12 tense split comparison, text-only LR, n_seeds={n_seeds}\n\n{table}"
    print("\n" + out)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG, "a") as fp:
        fp.write(f"\n## {ts}  H12 時制分割 比較 (n_seeds={n_seeds})\n\n```\n{out}\n```\n")
    print(f"\n-> {LOG} に追記")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
