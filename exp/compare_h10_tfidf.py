"""H10: Japanese char TF-IDF settings for `今後のDX展望`.

This is a text-only comparison.  It keeps CV splits, LogisticRegression, and
threshold search identical across candidates, so the only intended difference
is the TF-IDF analyzer/ngram setting.

Usage:
  python3 exp/compare_h10_tfidf.py [n_seeds]
"""
import datetime
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline


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


def normalize(s):
    """NFKC, number masking, and whitespace compression."""
    s = unicodedata.normalize("NFKC", str(s))
    s = _NUM.sub("0", s)
    s = _SPACE.sub(" ", s)
    return s.strip()


def build_model(analyzer, ngram_range):
    params = dict(BASE_TFIDF_PARAMS)
    params.update(analyzer=analyzer, ngram_range=ngram_range)
    return make_pipeline(TfidfVectorizer(**params),
                         LogisticRegression(**LR_PARAMS))


def best_f1(y, p):
    f1s = np.array([f1_score(y, p >= t) for t in THS])
    i = int(np.argmax(f1s))
    return float(f1s[i]), float(THS[i])


def _one_fold(texts, y, candidate, seed, fold, tr, va):
    model = build_model(candidate["analyzer"], candidate["ngram_range"])
    model.fit(texts[tr], y[tr])
    pred = model.predict_proba(texts[va])[:, 1]
    fold_f1, fold_th = best_f1(y[va], pred)
    return seed, fold, va, pred, fold_f1, fold_th


def evaluate_candidate(texts, y, candidate, seeds):
    jobs = []
    for seed in seeds:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            jobs.append((seed, fold, tr, va))

    out = Parallel(n_jobs=N_JOBS)(
        delayed(_one_fold)(texts, y, candidate, seed, fold, tr, va)
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
            analyzer=candidate["analyzer"],
            ngram=str(candidate["ngram_range"]),
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
                seed=seed,
                fold=fold,
                fold_f1=fold_f1,
                fold_th=fold_th,
            ))

    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def summarize(df):
    agg = df.groupby("name", sort=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        th_mean=("th", "mean"),
        th_std=("th", "std"),
    ).reset_index()
    return agg


def format_table(summary):
    lines = [
        "name                AUC             AP              F1              th",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"{r.name:18s}  "
            f"{r.auc_mean:.4f}±{r.auc_std:.4f}  "
            f"{r.ap_mean:.4f}±{r.ap_std:.4f}  "
            f"{r.f1_mean:.4f}±{r.f1_std:.4f}  "
            f"{r.th_mean:.3f}±{r.th_std:.3f}"
        )
    return "\n".join(lines)


def main(n_seeds=5):
    train = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    y = train[TARGET].values
    raw = train[TEXT_COL].fillna("").astype(str).values
    norm = np.array([normalize(x) for x in raw], dtype=object)

    candidates = [
        dict(name="H10-1 char_wb(1,3)", analyzer="char_wb", ngram_range=(1, 3)),
        dict(name="H10-2 char(1,3)", analyzer="char", ngram_range=(1, 3)),
        dict(name="H10-3 char(2,4)", analyzer="char", ngram_range=(2, 4)),
        dict(name="H10-4 char(2,6)", analyzer="char", ngram_range=(2, 6)),
        dict(name="H10-5 char(3,6)", analyzer="char", ngram_range=(3, 6)),
    ]
    seeds = list(range(n_seeds))

    all_rows = []
    all_fold_rows = []
    for c in candidates:
        print(f"--- {c['name']} ---", flush=True)
        rows, fold_rows = evaluate_candidate(raw, y, c, seeds)
        all_rows.append(rows.assign(text_version="raw"))
        all_fold_rows.append(fold_rows.assign(text_version="raw"))

    raw_df = pd.concat(all_rows, ignore_index=True)
    raw_fold_df = pd.concat(all_fold_rows, ignore_index=True)
    raw_summary = summarize(raw_df)

    # Normalized comparison is reported separately; H8 parity remains the raw run.
    norm_rows = []
    for c in candidates:
        print(f"--- normalized {c['name']} ---", flush=True)
        rows, _ = evaluate_candidate(norm, y, c, seeds)
        norm_rows.append(rows.assign(text_version="normalized"))
    norm_df = pd.concat(norm_rows, ignore_index=True)
    norm_summary = summarize(norm_df)

    raw_df.to_csv(os.path.join(EXP_DIR, "_h10_tfidf_raw_scores.csv"), index=False)
    raw_fold_df.to_csv(os.path.join(EXP_DIR, "_h10_tfidf_raw_fold_scores.csv"),
                       index=False)
    raw_summary.to_csv(os.path.join(EXP_DIR, "_h10_tfidf_raw_summary.csv"),
                       index=False)
    norm_df.to_csv(os.path.join(EXP_DIR, "_h10_tfidf_normalized_scores.csv"),
                   index=False)
    norm_summary.to_csv(os.path.join(EXP_DIR, "_h10_tfidf_normalized_summary.csv"),
                        index=False)

    raw_table = format_table(raw_summary)
    norm_table = format_table(norm_summary)
    out = (
        f"H10 TF-IDF comparison, text-only LR, n_seeds={n_seeds}\n\n"
        f"[raw: H8 parity]\n{raw_table}\n\n"
        f"[normalized: NFKC + number mask + whitespace]\n{norm_table}"
    )
    print("\n" + out)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG, "a") as fp:
        fp.write(f"\n## {ts}  H10 TF-IDF 比較 (n_seeds={n_seeds})\n\n```\n{out}\n```\n")
    print(f"\n-> {LOG} に追記")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
