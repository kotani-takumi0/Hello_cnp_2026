"""仮説を1つ検証する。

  python exp/run_hypothesis.py H1

現在の baseline に対し、同一seedでペア比較し判定を出す。
採用済み仮説は BASELINE_STACK に積んで新baselineとする（greedy forward）。
"""
import sys
import os
import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from harness import evaluate
from hypotheses import REGISTRY, FOLD_REGISTRY, NEEDS_CV
from decision import verdict, format_report, N_SEEDS

# 採用済み仮説をここに積む（例: ["H1", "H3"]）。現baselineの定義。
BASELINE_STACK = ["H1"]  # H1(利益率)採用済み→新baseline
LOG = os.path.join(os.path.dirname(__file__), "hypothesis_log.md")


def stack_adder(names):
    """複数の仮説特徴を順に適用する合成 add_features。空なら素のbaseline。"""
    if not names:
        return None
    funcs = [REGISTRY[n][1] for n in names]

    def _add(train, X):
        for f in funcs:
            X = f(train, X)
        return X
    return _add


def get_baseline_scores():
    """現baseline(=BASELINE_STACK)のper-seedスコア。stackが空ならキャッシュ利用。"""
    cache = os.path.join(os.path.dirname(__file__), "_baseline_scores.csv")
    if not BASELINE_STACK and os.path.exists(cache):
        df = pd.read_csv(cache).head(N_SEEDS).reset_index(drop=True)
        df.attrs["n_feat"] = 44
        return df  # キャッシュ行順=seed 0,1,.. なので先頭N_SEEDSで同一seedペアになる
    return evaluate(stack_adder(BASELINE_STACK), seeds=range(N_SEEDS),
                    label="baseline", verbose=False)


def main(name):
    if name in NEEDS_CV:
        print(f"{name} は未実装（{NEEDS_CV[name]}）。")
        return
    is_fold = name in FOLD_REGISTRY
    if name not in REGISTRY and not is_fold:
        print(f"未知の仮説: {name}. 選択肢: {list(REGISTRY) + list(FOLD_REGISTRY)}")
        return

    desc = (FOLD_REGISTRY if is_fold else REGISTRY)[name][0]
    ftrans = FOLD_REGISTRY[name][1] if is_fold else None
    base = get_baseline_scores()

    stack_key = "+".join(BASELINE_STACK + [name]) or name
    cand_cache = os.path.join(os.path.dirname(__file__), f"_cand_{stack_key}.csv")
    if os.path.exists(cand_cache):
        cand = pd.read_csv(cand_cache)
        cand.attrs["n_feat"] = "?"
        print(f"(候補スコアをキャッシュから再判定: {cand_cache})")
    else:
        # fold版は特徴をfold_transformで足す。通常版はadd_featuresに積む。
        add_names = BASELINE_STACK if is_fold else BASELINE_STACK + [name]
        cand = evaluate(stack_adder(add_names), seeds=range(N_SEEDS),
                        label=name, verbose=False, fold_transform=ftrans)
        cand.to_csv(cand_cache, index=False)

    v, d = verdict(base, cand)
    report = format_report(name, desc, base, cand, v, d)
    print(report)

    with open(LOG, "a") as fp:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        fp.write(f"\n## {ts}  baseline_stack={BASELINE_STACK}\n\n```\n{report}\n```\n")
    print(f"\n-> {LOG} に追記")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "H1")
