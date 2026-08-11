"""H9(LLM3軸) が H8(TF-IDFスタッキング) に対して何なのかを複数seedで決着させる。

Public は4構成とも 0.6838〜0.7040 に収まり区別がつかない（最大差0.0202＝H8/H8bで
ノイズと確定済みの幅そのもの）。OOF単seedの順位ともPublicの順位が食い違うため、
同一seedのペア比較で判定する。

問いは3つ:
  1. H9 は H8 に**上乗せ**になるか      -> H1+H8+H9 vs H1+H8
  2. H9 は H8 を**置き換え**られるか    -> H1+H9    vs H1+H8
  3. H9 単体に実体はあるか              -> H1+H9    vs H1

  python exp/compare_h9.py [n_seeds]
"""
import os
import sys
import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import evaluate
from hypotheses import REGISTRY, FOLD_REGISTRY
from decision import verdict, format_report

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "hypothesis_log.md")

# (表示名, add_features に積む仮説, fold_transform)
ARMS = [
    ("H1", ["H1"], None),
    ("H1+H8", ["H1"], "H8"),
    ("H1+H8+H9", ["H1", "H9"], "H8"),
    ("H1+H9", ["H1", "H9"], None),
]

# (候補, 基準, 何を問うているか)
QUESTIONS = [
    ("H1+H8", "H1", "H8の再確認（基準モデルの実体）"),
    ("H1+H9", "H1", "H9単体に実体はあるか"),
    ("H1+H8+H9", "H1+H8", "H9はH8への上乗せになるか"),
    ("H1+H9", "H1+H8", "H9はH8を置き換えられるか"),
]


def stack_adder(names):
    funcs = [REGISTRY[n][1] for n in names]

    def _add(train, X):
        for f in funcs:
            X = f(train, X)
        return X
    return _add


def main(n_seeds):
    seeds = range(n_seeds)
    runs = {}
    for name, adds, ft_key in ARMS:
        print(f"--- {name} ---", flush=True)
        ft = FOLD_REGISTRY[ft_key][1] if ft_key else None
        df = evaluate(stack_adder(adds), seeds=seeds, label=name, verbose=True,
                      fold_transform=ft)
        df.to_csv(os.path.join(HERE, f"_cmp9_{name.replace('+', '_')}.csv"), index=False)
        runs[name] = df

    reports = []
    for cand, base, desc in QUESTIONS:
        v, d = verdict(runs[base], runs[cand])
        reports.append(format_report(f"{cand} vs {base}", desc, runs[base],
                                     runs[cand], v, d))
    out = "\n\n".join(reports)
    print("\n" + out)

    with open(LOG, "a") as fp:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        fp.write(f"\n## {ts}  H9(LLM3軸) 比較 (n_seeds={n_seeds})\n\n```\n{out}\n```\n")
    print(f"\n-> {LOG} に追記")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
