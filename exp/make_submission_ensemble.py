"""Create an ensemble submission from repeated CV base models.

Default target configuration is the current best feature set:
  H1 + H8 + H9 + H13, with LGBM/CatBoost rank averaging.

CatBoost is optional at runtime. If it is not installed, `models.available`
will skip it and the script still produces the LGBM-only reference output.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend import blend, rank_correlation, seed_metrics
from ensemble_cv import run_repeated_cv, save
from threshold import analyze, format_report, rate_match_cut


def _tag(models, use_llm, use_org_chart, use_overview, use_dx_outlook_manual,
         use_company_overview_manual, method, n_seeds):
    parts = ["ensemble"] + list(models)
    parts += ["h8b" if use_overview else "h8"]
    if use_llm:
        parts.append("h9")
    if use_org_chart:
        parts.append("h13")
    if use_dx_outlook_manual:
        parts.append("h14")
    if use_company_overview_manual:
        parts.append("h16")
    parts += [method, f"{n_seeds}seed"]
    return "_".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lgbm", "catboost"],
                   help="Base models. Choices currently include lgbm catboost.")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--method", choices=["rank", "prob"], default="rank")
    p.add_argument("--llm", action="store_true", help="Add H9 LLM structured features.")
    p.add_argument("--org-chart", action="store_true", help="Add H13 organization features.")
    p.add_argument("--dx-outlook-manual", action="store_true",
                   help="Add H14 manual DX outlook features.")
    p.add_argument("--company-overview-manual", action="store_true",
                   help="Add H16 manual company overview features.")
    p.add_argument("--overview", action="store_true", help="Use H8b text input.")
    p.add_argument("--no-text", action="store_true", help="Disable H8/H8b text stacking.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-tag", default=None)
    p.add_argument("--save-npz", default=None)
    args = p.parse_args()

    res = run_repeated_cv(
        n_seeds=args.n_seeds,
        models=tuple(args.models),
        use_text=not args.no_text,
        use_overview=args.overview,
        use_llm=args.llm,
        use_org_chart=args.org_chart,
        use_dx_outlook_manual=args.dx_outlook_manual,
        use_company_overview_manual=args.company_overview_manual,
        data_dir=args.data_dir,
        verbose=True,
    )
    if args.save_npz:
        save(res, args.save_npz)

    models = res["models"]
    y = res["y"]
    oof_parts = [res["OOF"][m] for m in models]
    test_parts = [res["TEST"][m] for m in models]
    B_oof = blend(oof_parts, method=args.method)
    B_test = blend(test_parts, method=args.method)

    print("\n=== base model metrics ===")
    for m in models:
        sm = seed_metrics(y, res["OOF"][m], res["FOLD"])
        print(f"{m:8s} AUC {sm['auc'].mean():.4f} AP {sm['ap'].mean():.4f} "
              f"F1 {sm['f1'].mean():.4f} th {sm['th'].mean():.3f}")

    if len(models) == 2:
        print(f"rank corr({models[0]}, {models[1]}): "
              f"{rank_correlation(res['OOF'][models[0]], res['OOF'][models[1]]):.4f}")

    print("\n=== ensemble metrics ===")
    sm = seed_metrics(y, B_oof, res["FOLD"])
    print(f"AUC {sm['auc'].mean():.4f} AP {sm['ap'].mean():.4f} "
          f"F1 {sm['f1'].mean():.4f} th {sm['th'].mean():.3f}")
    a = analyze(y, B_oof)
    print(format_report(a, label=f"{'+'.join(models)} {args.method}"))

    final_score = B_test.mean(axis=0)
    cut = rate_match_cut(final_score, a["rate"])
    label = (final_score >= cut).astype(int)
    print(f"test cut by rate match: {cut:.6f}")
    print(f"test positive rate: {label.mean():.4f} / train target rate {y.mean():.4f}")
    print(f"sanity AUC/AP on mean OOF: "
          f"{roc_auc_score(y, B_oof.mean(axis=0)):.4f} / "
          f"{average_precision_score(y, B_oof.mean(axis=0)):.4f}")

    sample = pd.read_csv(f"{args.data_dir}/sample_submit.csv", header=None,
                         names=["企業ID", "購入フラグ"])
    pred_df = pd.DataFrame({"企業ID": res["test_ids"], "pred": label})
    sub = sample[["企業ID"]].merge(pred_df, on="企業ID", how="left")
    assert sub["pred"].notna().all(), "予測欠損あり"
    sub["pred"] = sub["pred"].astype(int)

    tag = args.out_tag or _tag(models, args.llm, args.org_chart, args.overview,
                               args.dx_outlook_manual,
                               args.company_overview_manual,
                               args.method, args.n_seeds)
    out = f"submission/submission_{tag}.csv"
    os.makedirs("submission", exist_ok=True)
    sub.to_csv(out, index=False, header=False, lineterminator="\n")
    print(f"保存: {out} 行数={len(sub)} 正例={int(sub['pred'].sum())}")


if __name__ == "__main__":
    main()
