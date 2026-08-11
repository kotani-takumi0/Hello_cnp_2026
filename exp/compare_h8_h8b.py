"""H8 (今後のDX展望のみ) vs H8b (企業概要+今後のDX展望) を複数seedで決着させる。

単seed OOF F1 では H8 が上(0.7500 vs 0.7236)だったが Public は H8b が上
(0.7040 vs 0.6838)。差はいずれもノイズ幅に近いため、同一seedのペア比較で判定する。

  python exp/compare_h8_h8b.py [n_seeds]
"""
import os
import sys
import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import evaluate
from hypotheses import REGISTRY, FOLD_REGISTRY
from decision import verdict, format_report

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hypothesis_log.md")


def main(n_seeds):
    seeds = range(n_seeds)
    h1 = REGISTRY["H1"][1]
    runs = {}
    for name, ft in [("baseline(H1)", None),
                     ("H1+H8", FOLD_REGISTRY["H8"][1]),
                     ("H1+H8b", FOLD_REGISTRY["H8b"][1])]:
        print(f"--- {name} ---", flush=True)
        df = evaluate(h1, seeds=seeds, label=name, verbose=True, fold_transform=ft)
        df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"_cmp_{name.replace('+', '_')}.csv"), index=False)
        runs[name] = df

    reports = []
    for cand in ["H1+H8", "H1+H8b"]:
        v, d = verdict(runs["baseline(H1)"], runs[cand])
        reports.append(format_report(cand, "vs baseline(H1)", runs["baseline(H1)"],
                                     runs[cand], v, d))
    # 本題: H8 を baseline 扱いにして H8b の増分を見る
    v, d = verdict(runs["H1+H8"], runs["H1+H8b"])
    reports.append(format_report("H8b vs H8", "企業概要の追加が効くか", runs["H1+H8"],
                                 runs["H1+H8b"], v, d))
    out = "\n\n".join(reports)
    print("\n" + out)

    with open(LOG, "a") as fp:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        fp.write(f"\n## {ts}  H8/H8b 比較 (n_seeds={n_seeds})\n\n```\n{out}\n```\n")
    print(f"\n-> {LOG} に追記")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
