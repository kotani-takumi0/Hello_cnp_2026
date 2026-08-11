"""Compare H8 with H12 tense-aware text stacking.

H12 keeps the full-text H8 score and adds current/future text scores.  This is
different from `compare_h12_tense.py`, which is only a text-only diagnostic.

Usage:
  python3 exp/compare_h12_stack.py [n_seeds]
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import format_report, verdict
from harness import evaluate
from hypotheses import FOLD_REGISTRY, REGISTRY


LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hypothesis_log.md")


def main(n_seeds=10):
    seeds = range(n_seeds)
    h1 = REGISTRY["H1"][1]
    models = [
        ("baseline(H1)", None),
        ("H1+H8", FOLD_REGISTRY["H8"][1]),
        ("H1+H12fc", FOLD_REGISTRY["H12fc"][1]),
        ("H1+H12", FOLD_REGISTRY["H12"][1]),
    ]

    runs = {}
    for name, ft in models:
        print(f"--- {name} ---", flush=True)
        df = evaluate(h1, seeds=seeds, label=name, verbose=True,
                      fold_transform=ft)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"_cmp_{name.replace('+', '_')}.csv")
        df.to_csv(out, index=False)
        runs[name] = df

    reports = []
    for cand in ["H1+H8", "H1+H12fc", "H1+H12"]:
        v, d = verdict(runs["baseline(H1)"], runs[cand])
        reports.append(format_report(cand, "vs baseline(H1)",
                                     runs["baseline(H1)"], runs[cand], v, d))
    for cand in ["H1+H12fc", "H1+H12"]:
        v, d = verdict(runs["H1+H8"], runs[cand])
        reports.append(format_report(cand, "vs H1+H8",
                                     runs["H1+H8"], runs[cand], v, d))

    out = "\n\n".join(reports)
    print("\n" + out)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG, "a") as fp:
        fp.write(f"\n## {ts}  H12 スタッキング比較 (n_seeds={n_seeds})\n\n```\n{out}\n```\n")
    print(f"\n-> {LOG} に追記")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
