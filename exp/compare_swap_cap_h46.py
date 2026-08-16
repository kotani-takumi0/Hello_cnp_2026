"""H46: H44 R4 rank blendに交換数上限を設けたnested評価。"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_anchored_r4 import _best_threshold  # noqa: E402
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402

INP = Path("exp/_r4_modern_nca_discovery_seed20260815.npz")
OUT_NPZ = Path("exp/_h46_swap_cap_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_h46_swap_cap_discovery_seed20260815.json")
LAMBDAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
SWAPS = (0, 1, 2, 3, 4, 5, 8, 10, 12)
AP_TOLERANCE = 0.002
MIN_F1_GAIN = 0.005
MAX_FOLD_DROP = 0.02

def rank(x): return rankdata(np.asarray(x, float), method="average") / len(x)
def support(base, r4, lam): return (1-lam)*rank(base) + lam*rank(r4)

def swap_labels(base, s, th, m):
    b = (base >= th).astype(int)
    if m == 0: return b
    neg = np.flatnonzero(b == 0); pos = np.flatnonzero(b == 1)
    m = min(m, len(neg), len(pos))
    out = b.copy()
    out[neg[np.argsort(-s[neg], kind="mergesort")[:m]]] = 1
    out[pos[np.argsort(s[pos], kind="mergesort")[:m]]] = 0
    assert out.sum() == b.sum()
    return out

def select(y, base, r4, seed):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    rows=[]
    for lam in LAMBDAS:
        for m in SWAPS:
            aps=[]; fs=[]; bfs=[]
            for tr,va in inner.split(np.zeros(len(y)),y):
                th,_=_best_threshold(y[tr],base[tr]); b=(base[va]>=th).astype(int)
                s=support(base[va],r4[va],lam); c=swap_labels(base[va],s,th,m)
                aps.append(average_precision_score(y[va],s)); fs.append(f1_score(y[va],c)); bfs.append(f1_score(y[va],b))
            rows.append(dict(lam=lam,m=m,ap=float(np.mean(aps)),f1=float(np.mean(fs)),bf1=float(np.mean(bfs)),aps=aps,fs=fs))
    base_ap=max(r["ap"] for r in rows if r["lam"]==0)
    for r in rows: r["ap_delta"]=r["ap"]-base_ap; r["f1_delta"]=r["f1"]-r["bf1"]; r["eligible"]=r["ap_delta"]>=-AP_TOLERANCE
    return max([r for r in rows if r["eligible"]],key=lambda r:(r["f1"],r["ap"],-r["m"],-r["lam"])),rows

def run(smoke=False):
    z=np.load(INP); y=z["y"].astype(int); base=z["base_blend"].astype(float); r4=z["modern_nca_oof"].astype(float)
    folds=StratifiedKFold(INNER_FOLDS,shuffle=True,random_state=LOCKBOX_SEED)
    co=np.full(len(y),np.nan); bl=np.full(len(y),-1,int); cl=np.full(len(y),-1,int); ds=[]
    for fold,(tr,va) in enumerate(folds.split(np.zeros(len(y)),y),1):
        if smoke and fold>1: break
        choice,grid=select(y[tr],base[tr],r4[tr],LOCKBOX_SEED+fold*10000)
        th,_=_best_threshold(y[tr],base[tr]); b=(base[va]>=th).astype(int); s=support(base[va],r4[va],choice['lam']); c=swap_labels(base[va],s,th,choice['m'])
        co[va]=s; bl[va]=b; cl[va]=c
        row=dict(fold=fold,threshold=th,**choice,positive_budget=int(b.sum()),swap_actual=int(((b==0)&(c==1)).sum()),outer_base_ap=float(average_precision_score(y[va],base[va])),outer_candidate_ap=float(average_precision_score(y[va],s)),outer_base_f1=float(f1_score(y[va],b)),outer_candidate_f1=float(f1_score(y[va],c)),grid=grid)
        ds.append(row); print(f"fold {fold}: lam={choice['lam']:.2f} m={choice['m']} K={b.sum()} swaps={row['swap_actual']} ΔAP={row['outer_candidate_ap']-row['outer_base_ap']:+.4f} ΔF1={row['outer_candidate_f1']-row['outer_base_f1']:+.4f}")
    if smoke:
        done=np.isfinite(co); assert done.sum()>0 and np.array_equal(bl[done].sum(),cl[done].sum()); print('SMOKE OK'); return
    ba=average_precision_score(y,base); ca=average_precision_score(y,co); bf=f1_score(y,bl); cf=f1_score(y,cl); fa=[d['outer_candidate_f1']-d['outer_base_f1'] for d in ds]
    err=dict(base_errors=int((bl!=y).sum()),candidate_errors=int((cl!=y).sum()),fn_rescued=int(((y==1)&(bl==0)&(cl==1)).sum()),new_fn=int(((y==1)&(bl==1)&(cl==0)).sum()),fp_removed=int(((y==0)&(bl==1)&(cl==0)).sum()),new_fp=int(((y==0)&(bl==0)&(cl==1)).sum()),zero_to_one=int(((bl==0)&(cl==1)).sum()),one_to_zero=int(((bl==1)&(cl==0)).sum()))
    verdict='PROMISING' if ca-ba>=-AP_TOLERANCE and cf-bf>=MIN_F1_GAIN and min(fa)>=-MAX_FOLD_DROP else 'REJECT'
    print('\n=== H46 swap cap ==='); print(pd.DataFrame({'exp032':[roc_auc_score(y,base),ba,bf],'H46':[roc_auc_score(y,co),ca,cf]},index=['AUC','AP','budgeted F1']).to_string(float_format=lambda x:f'{x:.4f}')); print(f'ΔAP={ca-ba:+.4f} ΔF1={cf-bf:+.4f} foldΔF1={[round(x,4) for x in fa]} errors={json.dumps(err,ensure_ascii=False)} verdict={verdict}')
    result=dict(method='H44 R4 rank blend with fixed-K swap cap',lambda_grid=list(LAMBDAS),swap_grid=list(SWAPS),scores={'exp032':{'auc':roc_auc_score(y,base),'ap':ba,'f1':bf},'H46':{'auc':roc_auc_score(y,co),'ap':ca,'f1':cf}},deltas={'ap':ca-ba,'f1':cf-bf},fold_f1_deltas=fa,errors=err,diagnostics=ds,verdict=verdict,lockbox_opened=False)
    np.savez_compressed(OUT_NPZ,y=y,exp032=base,candidate=co,exp032_label=bl,candidate_label=cl); OUT_JSON.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'保存: {OUT_NPZ} / {OUT_JSON}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); run(p.parse_args().smoke)
