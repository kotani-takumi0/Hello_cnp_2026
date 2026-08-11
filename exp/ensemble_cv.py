"""複数ベースモデルの Repeated CV ランナー。

1 seed = 外側 StratifiedKFold(5)。**全モデルが同じ fold 分割・同じ特徴量**を見るので、
出てきた OOF は行単位で揃う。これが揃っていないとペア比較もブレンドも成立しない。

計算量の勘所: テキストスタッキング（ネストCV: 内側5fold + full fit = 6回のTF-IDF+LR）は
外側 fold ごとに **1回だけ** 計算して全モデルで共有する。テキストモデルは下流のGBDTに
依存しないので、モデルを2本に増やしても増えるのはGBDTの学習時間だけ。

出力は「指標」ではなく **OOF/test の確率行列そのもの**。
  OOF[model]  : (n_seeds, n_train)
  TEST[model] : (n_seeds, n_test)
指標も閾値もブレンドも、後からこの行列だけで再計算できる。重い実行を Colab に投げて
npz を持ち帰り、手元では解析だけ回す、という分業のため。
"""
import os
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_frames, build_matrices, build_texts, build_org_texts, ID_COL
from models import MODEL_REGISTRY, available
from text_features import fold_text_preds, INNER_SEED
from organization_features import ORG_SCORE_COL, fold_org_preds

N_SPLITS = 5


def run_seed(X, y, Xte, txt_tr, txt_te, feat_name, models, seed,
             use_text=True, org_tr=None, org_te=None, use_org_chart=False,
             vary_inner_seed=False):
    """1 seed 分の (モデル名 -> (oof, test_pred)), fold_id, best_iter を返す。"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    inner_seed = seed if vary_inner_seed else INNER_SEED

    oof = {m: np.zeros(len(X)) for m in models}
    te_folds = {m: [] for m in models}
    gains = {m: [] for m in models}
    best_iters = {m: [] for m in models}
    fold_id = np.zeros(len(X), dtype=int)

    for k, (tr, va) in enumerate(skf.split(X, y)):
        fold_id[va] = k
        Xtr, Xva, Xte_f = X.iloc[tr].copy(), X.iloc[va].copy(), Xte.copy()

        if use_text:
            # ここが fold あたり1回。全ベースモデルで共有する。
            t_tr, t_va, t_te = fold_text_preds(txt_tr[tr], y[tr], txt_tr[va],
                                               txt_te, inner_seed)
            Xtr[feat_name], Xva[feat_name], Xte_f[feat_name] = t_tr, t_va, t_te
        if use_org_chart:
            o_tr, o_va, o_te = fold_org_preds(org_tr[tr], y[tr], org_tr[va],
                                              org_te, inner_seed)
            Xtr[ORG_SCORE_COL], Xva[ORG_SCORE_COL], Xte_f[ORG_SCORE_COL] = \
                o_tr, o_va, o_te

        for name in models:
            _, fn = MODEL_REGISTRY[name]
            r = fn(Xtr, y[tr], Xva, y[va], Xte_f, seed)
            oof[name][va] = r["va"]
            te_folds[name].append(r["te"])
            gains[name].append(r["gain"])
            best_iters[name].append(r["best_iter"])

    test_pred = {m: np.mean(te_folds[m], axis=0) for m in models}
    return oof, test_pred, fold_id, gains, best_iters


def run_repeated_cv(n_seeds=20, models=("lgbm", "catboost"), use_text=True,
                    use_overview=False, use_llm=False, data_dir="data",
                    use_org_chart=False, use_dx_outlook_manual=False,
                    use_company_overview_manual=False,
                    vary_inner_seed=False, verbose=True):
    """Repeated CV を回して確率行列を返す。"""
    models = available(list(models))
    train, test = load_frames(data_dir)
    X, y, Xte = build_matrices(train, test, use_llm=use_llm,
                               use_org_chart=use_org_chart,
                               use_dx_outlook_manual=use_dx_outlook_manual,
                               use_company_overview_manual=use_company_overview_manual,
                               data_dir=data_dir)
    txt_tr, txt_te, feat_name = build_texts(train, test, use_overview=use_overview)
    org_tr, org_te = build_org_texts(train, test) if use_org_chart else (None, None)

    if verbose:
        print(f"models={models} n_seeds={n_seeds} use_text={use_text} "
              f"use_llm={use_llm} use_overview={use_overview} "
              f"use_org_chart={use_org_chart} "
              f"use_dx_outlook_manual={use_dx_outlook_manual} "
              f"use_company_overview_manual={use_company_overview_manual}")
        n_fold_feat = int(use_text) + int(use_org_chart)
        print(f"X={X.shape} Xte={Xte.shape} 正例率={y.mean():.4f} "
              f"n_feat={X.shape[1] + n_fold_feat}")

    OOF = {m: np.zeros((n_seeds, len(X))) for m in models}
    TEST = {m: np.zeros((n_seeds, len(Xte))) for m in models}
    FOLD = np.zeros((n_seeds, len(X)), dtype=int)
    gain_acc = {m: [] for m in models}
    iter_acc = {m: [] for m in models}

    t0 = time.time()
    for s in range(n_seeds):
        ts = time.time()
        oof, tep, fid, gains, iters = run_seed(
            X, y, Xte, txt_tr, txt_te, feat_name, models, s,
            use_text=use_text, org_tr=org_tr, org_te=org_te,
            use_org_chart=use_org_chart, vary_inner_seed=vary_inner_seed)
        FOLD[s] = fid
        for m in models:
            OOF[m][s], TEST[m][s] = oof[m], tep[m]
            gain_acc[m].extend(gains[m])
            iter_acc[m].extend(iters[m])
        if verbose:
            from sklearn.metrics import roc_auc_score, average_precision_score
            msg = "  ".join(
                f"{m}: AUC {roc_auc_score(y, oof[m]):.4f} "
                f"AP {average_precision_score(y, oof[m]):.4f}" for m in models)
            print(f"seed {s:2d}  {msg}   ({time.time()-ts:.0f}s / "
                  f"累計 {time.time()-t0:.0f}s)", flush=True)

    return dict(OOF=OOF, TEST=TEST, FOLD=FOLD, y=y,
                test_ids=test[ID_COL].values, models=models,
                feature_names=list(X.columns) + ([feat_name] if use_text else [])
                + ([ORG_SCORE_COL] if use_org_chart else []),
                gains=gain_acc, best_iters=iter_acc,
                config=dict(n_seeds=n_seeds, models=models, use_text=use_text,
                            use_overview=use_overview, use_llm=use_llm,
                            use_org_chart=use_org_chart,
                            use_dx_outlook_manual=use_dx_outlook_manual,
                            use_company_overview_manual=use_company_overview_manual,
                            vary_inner_seed=vary_inner_seed))


def save(res, path):
    """npz に保存（解析側は fit なしでこれだけ読めば良い）。"""
    import json
    arrays = dict(FOLD=res["FOLD"], y=res["y"], test_ids=res["test_ids"])
    for m in res["models"]:
        arrays[f"OOF__{m}"] = res["OOF"][m]
        arrays[f"TEST__{m}"] = res["TEST"][m]
    arrays["meta"] = np.array(json.dumps(
        dict(config=res["config"], models=res["models"],
             feature_names=res["feature_names"]), ensure_ascii=False))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"保存: {path}")


def load(path):
    import json
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    models = meta["models"]
    return dict(OOF={m: z[f"OOF__{m}"] for m in models},
                TEST={m: z[f"TEST__{m}"] for m in models},
                FOLD=z["FOLD"], y=z["y"], test_ids=z["test_ids"],
                models=models, feature_names=meta["feature_names"],
                config=meta["config"])
