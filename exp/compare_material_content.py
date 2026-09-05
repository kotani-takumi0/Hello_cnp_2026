"""H96: what is *in* the material, and does "out of scope" explain the misses?

H92 tested the same hypothesis one way: it wrote down what the product is
(an e-learning platform + a skill assessment tool), turned that into keyword
and semantic scope-fit scores, and asked whether the model's FPs score low.
Four instruments said no.  That test has two assumptions baked in, and this
file removes them one at a time.

Assumption 1 -- the product description came from the competition brief and
from my reading, not from the data.  ``content`` inverts the problem: give the
encoder a *menu* of twelve things a DX training product could contain, ask
which content each company is asking for, and let the purchase label say which
menu item this product actually is.  Similarities are row-centred, so the
"how eager are they" axis -- the one every text feature in this project ends
up measuring -- cancels before anything is scored.

Assumption 2 -- the rows under test were defined by the *composite* p>0.5, and
the composite is driven by the tabular experts.  H92 4.1 found the text expert
gives FP and TP the same logit, i.e. the rows it examined were never "the text
says buy".  ``premise`` re-conditions on the text alone: take the companies the
DX-outlook expert ranks highest and ask what happens to them.  That group is
where the user's hypothesis actually lives.

``pocket`` then tests the hypothesis in the only form that would make it
operational.  "Out of scope" is not a correlation, it is a *necessary
condition*: somewhere in the data there should be a region whose purchase rate
collapses regardless of enthusiasm.  Rather than propose regions by hand, this
enumerates every single condition and every pair over a label-free bank and
reports the best rejection rule found, against a null that repeats the entire
search on shuffled labels -- so the multiplicity of the search is priced in.
The bar is the one derived in h92 9.2: a rejection rule needs precision
``1 - F1/2`` to pay for itself.

``screen`` puts whatever the descriptive modes turn up through the production
judge: eleven seeds, ninth expert, paired against the same seeds' baseline.

Run:
  .venv/bin/python exp/compare_material_content.py --mode premise
  .venv/bin/python exp/compare_material_content.py --mode content   # e5, ~60s
  .venv/bin/python exp/compare_material_content.py --mode pocket
  .venv/bin/python exp/compare_material_content.py --mode screen
  .venv/bin/python exp/compare_material_content.py --mode gate --min-support 45
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_material_scope import (  # noqa: E402
    IN_SCOPE, OUT_SCOPE, build_material_scope_features, education_context,
)
from compare_short_term_debt_gate import _blend_oof_p  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, EXPERTS, _load  # noqa: E402
from ensemble_experts import THS, blend_oof  # noqa: E402

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
TEXT_COL = "今後のDX展望"
E5_MODEL = "intfloat/multilingual-e5-base"
SIM_CACHE = "exp/_h96_content_sims.npz"
TEXT_EXPERT = "E3_dx_text"
HOT_FRAC = 0.20          # 「テキストが買いたそう」＝テキストExpertの上位20%
CAT_COLS = ("業界", "上場種別", "特徴")

# ---------------------------------------------------------------------------
# 商材の中身の「メニュー」。この12枝から、購入ラベルに選ばせる。
# 2列目は計器検証用のキーワード（この類似度が本当にその話題を見ているかの確認）。
# ---------------------------------------------------------------------------
CONTENTS = {
    "AI・データ分析": (
        "AIや機械学習を業務で使いこなし、データを分析して意思決定に活かすスキルを学ぶ",
        "AI|人工知能|機械学習|データ分析|データサイエン|生成AI|アナリティクス"),
    "データ活用基盤": (
        "社内に散らばるデータを統合し、可視化してデータに基づく経営判断ができるようにする",
        "データ基盤|データ活用|可視化|ダッシュボード|BI|データベース|蓄積"),
    "クラウド・IT基盤": (
        "クラウドサービスやシステム基盤の仕組みを理解し、老朽化したITインフラを刷新する",
        "クラウド|インフラ|基幹システム|システム刷新|レガシー|SaaS|老朽"),
    "業務自動化・RPA": (
        "RPAやノーコードツールを使って定型業務を自動化し、事務作業を効率化する",
        "RPA|自動化|ノーコード|ローコード|ペーパーレス|電子化|効率化"),
    "セキュリティ": (
        "情報セキュリティの脅威を理解し、全社で情報を安全に取り扱えるようにする",
        "セキュリティ|情報漏|サイバー|安全"),
    "全社ITリテラシー": (
        "全社員がPCやデジタルツールを使いこなすための基礎的なITリテラシーを底上げする",
        "リテラシー|基礎|全社員|底上げ|ITパスポート|入門"),
    "DX戦略・経営": (
        "経営層や管理職がDXの戦略を描き、全社の変革をマネジメントできるようにする",
        "経営|戦略|マネジメント|管理職|経営層|ロードマップ|ガバナンス"),
    "業務プロセス改革": (
        "業務プロセスそのものを見直し、部門をまたいだ業務改革を進める",
        "業務プロセス|BPR|業務改革|標準化|見直し|効率"),
    "エンジニア育成": (
        "プログラミングやシステム開発ができる技術者を社内で育てて内製化する",
        "プログラミング|開発者|エンジニア|内製|システム開発|技術者"),
    "顧客接点デジタル化": (
        "ECやデジタルマーケティングによって顧客との接点をデジタル化する",
        "EC|顧客|マーケティング|オンライン販売|CRM|顧客接点|販売"),
    "現場IoT・設備": (
        "工場や店舗の現場にIoTを導入し、設備や作業のデジタル化を進める",
        "IoT|工場|生産|現場|設備|センサ|店舗|物流"),
    "DX推進リーダー": (
        "DXを社内で推進するリーダーや推進担当者を育成し、推進体制を作る",
        "推進|リーダー|中核|けん引|専門人材|DX人材|体制"),
}

# 事務職比率のprior。業種の就業構造だけで決めており、ラベルも購入率も見ていない。
# e-learning は「机の前にいる従業員」にしか配れない、という商材側の制約の代理。
DESK_SHARE = {
    "IT": 0.95, "通信": 0.90, "通信機器": 0.70, "金融": 0.90, "コンサルティング": 0.95,
    "人材": 0.85, "広告": 0.90, "マスコミ": 0.80, "ゲーム": 0.90, "商社": 0.80,
    "不動産": 0.70, "教育": 0.50, "専門サービス": 0.80, "その他サービス": 0.50,
    "機械関連サービス": 0.40, "機械": 0.35, "製造": 0.30, "化学": 0.35,
    "自動車・乗り物": 0.30, "電気製品": 0.35, "食品": 0.25, "生活用品": 0.30,
    "エネルギー": 0.40, "建設・工事": 0.30, "運輸・物流": 0.20, "小売": 0.15,
    "外食": 0.10, "アパレル・美容": 0.20, "医療・福祉": 0.20, "エンタメ": 0.30,
    "その他": 0.50,
}
STRUCT_COLS = ("店舗比率", "工場比率", "現場拠点比率", "log拠点あたり従業員",
               "log店舗あたり従業員", "デスク比率prior")


# ---------------------------------------------------------------------------
# 共通の下ごしらえ
# ---------------------------------------------------------------------------
def _logit(v):
    v = np.clip(np.asarray(v, float), 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def build_structure_features(raw):
    """商材が配れる相手か、を構造だけで測る。ラベル非依存。"""
    site = (raw[["事業所数", "工場数", "店舗数"]]
            .apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0))
    total = site.sum(axis=1)
    emp = pd.to_numeric(raw["従業員数"], errors="coerce").fillna(0).clip(lower=1)
    return pd.DataFrame({
        "店舗比率": site["店舗数"] / (total + 1),
        "工場比率": site["工場数"] / (total + 1),
        "現場拠点比率": (site["店舗数"] + site["工場数"]) / (total + 1),
        "log拠点あたり従業員": np.log1p(emp / (total + 1)),
        "log店舗あたり従業員": np.log1p(emp / (site["店舗数"] + 1)),
        "デスク比率prior": raw["業界"].map(DESK_SHARE).fillna(0.5).astype(float),
    }, index=raw.index)


def _blend_and_text(seed, config):
    y, Z, _ = _load(CONFIGS[config], seed)
    p = _blend_oof_p(y, Z, seed, "auto")
    cache = np.load(CONFIGS[config].format(seed=seed), allow_pickle=True)
    return y, p, cache


def _hot_mask(text_oof, frac=HOT_FRAC):
    """テキストExpertの上位frac。『DX展望が買いたそう』の定義をテキストだけで作る。"""
    k = int(round(len(text_oof) * frac))
    m = np.zeros(len(text_oof), bool)
    m[np.argsort(-text_oof)[:k]] = True
    return m


def _crossfit(X, y, seed, C=0.3, Xte=None):
    X = np.asarray(X, float)
    o = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=4000)).fit(X[tr], y[tr])
        o[va] = m.predict_proba(X[va])[:, 1]
    if Xte is None:
        return o
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(C=C, max_iter=4000)).fit(X, y)
    return o, m.predict_proba(np.asarray(Xte, float))[:, 1]


def _resid_auc(o, p, y):
    """合成pを除いた後の残差AUC。h92 9.4 の基準（0.55が損益分岐）。"""
    pl, ol = _logit(p).reshape(-1, 1), _logit(o)
    fit = LinearRegression().fit(pl, ol).predict(pl)
    return roc_auc_score(y, ol - fit)


def _rate_ci(y_sub):
    n = len(y_sub)
    r = float(np.mean(y_sub)) if n else float("nan")
    return r, 1.96 * np.sqrt(max(r * (1 - r), 1e-9) / max(n, 1))


# ---------------------------------------------------------------------------
# mode: premise -- 「テキストが買いたそう」をテキストだけで定義し直す
# ---------------------------------------------------------------------------
def run_premise(seeds, config):
    raw = pd.read_csv("data/train.csv")
    struct = build_structure_features(raw)

    print("=== 1. 『DX展望が買いたそう』な行を、テキストだけで定義する ===")
    print(f"  定義: {TEXT_EXPERT} のOOF確率 上位{HOT_FRAC:.0%}"
          f"（合成pは使わない。合成pは表形式Expertが動かしている）\n")
    print(f"{'seed':>5}{'熱心n':>7}{'買う':>6}{'買わない':>9}"
          f"{'  そのうち モデルも低評価':>22}{'  モデルが外す(FP)':>18}")
    tot = np.zeros(4)
    for seed in seeds:
        y, p, cache = _blend_and_text(seed, config)
        hot = _hot_mask(cache[f"oof_{TEXT_EXPERT}"])
        no = hot & (y == 0)
        ok, fp = int((no & (p <= 0.5)).sum()), int((no & (p > 0.5)).sum())
        print(f"{seed:>5}{int(hot.sum()):>7}{int((hot & (y == 1)).sum()):>6}"
              f"{int(no.sum()):>9}{ok:>22}{fp:>18}")
        tot += [hot.sum(), (hot & (y == 1)).sum(), no.sum(), fp]
    tot /= len(seeds)
    print(f"{'平均':>5}{tot[0]:>7.0f}{tot[1]:>6.0f}{tot[2]:>9.0f}"
          f"{tot[2] - tot[3]:>22.0f}{tot[3]:>18.0f}")
    print(f"\n  → 『テキストは買いたそうなのに買わない』は {tot[2]:.0f}行 実在する。"
          f"\n    だが {tot[2] - tot[3]:.0f}行はモデルが既に正しく低く見ており、"
          f"外すのは {tot[3]:.0f}行（全742行の{tot[3] / 742:.1%}）。"
          "\n    仮説が説明できる余地は、まずこの大きさしかない。")

    print("\n=== 2. では熱心群の中で、買う/買わないを分けているのは何か ===")
    print("  （＝商材の『範囲』をデータに語らせる。11seed平均の群内AUC）\n")
    rows = {n: [] for n in EXPERTS}
    rows["合成p"] = []
    for seed in seeds:
        y, p, cache = _blend_and_text(seed, config)
        hot = _hot_mask(cache[f"oof_{TEXT_EXPERT}"])
        for n in EXPERTS:
            rows[n].append(roc_auc_score(y[hot], cache["oof_" + n][hot]))
        rows["合成p"].append(roc_auc_score(y[hot], p[hot]))
    print(f"{'Expert':<14}{'熱心群内AUC':>12}")
    for n, v in sorted(rows.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{n:<14}{np.mean(v):>12.3f}")
    print(f"\n  {TEXT_EXPERT} が群内で無力なのは条件付けの当然の結果。"
          "読むべきは、代わりに何が効いているか。")

    y0, p0, cache0 = _blend_and_text(seeds[-1], config)
    hot0 = _hot_mask(cache0[f"oof_{TEXT_EXPERT}"])
    num = raw.select_dtypes(include=[np.number]).drop(columns=["企業ID", "購入フラグ"])
    out = []
    for c in num.columns:
        v = num[c].values.astype(float)
        ok = ~np.isnan(v)
        aucs = []
        for seed in seeds:
            y, _, cache = _blend_and_text(seed, config)
            m = _hot_mask(cache[f"oof_{TEXT_EXPERT}"]) & ok
            if m.sum() >= 100 and len(np.unique(y[m])) > 1:
                aucs.append(roc_auc_score(y[m], v[m]))
        if aucs:
            out.append((c, float(np.mean(aucs)), roc_auc_score(y0[ok], v[ok])))
    o = pd.DataFrame(out, columns=["列", "熱心群内AUC", "全体AUC"])
    o["ずれ"] = (o["熱心群内AUC"] - 0.5).abs()
    print("\n  生の列（熱心群内AUC、|0.5からの差|順の上位10）")
    print(o.sort_values("ずれ", ascending=False).head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== 3. 熱心群の購入率を業種で割る（商材が配れる相手かどうか）===")
    t = (pd.DataFrame({"業界": raw["業界"].values, "y": y0})[hot0]
         .groupby("業界").agg(n=("y", "size"), 熱心群購入率=("y", "mean")))
    allrate = pd.DataFrame({"業界": raw["業界"].values, "y": y0}).groupby(
        "業界")["y"].mean().rename("全体購入率")
    t = t.join(allrate).join(
        pd.Series(DESK_SHARE, name="デスク比率prior"))
    t = t[t.n >= 5].sort_values("熱心群購入率")
    print(t.round(3).to_string())
    g = (pd.DataFrame({"業界": raw["業界"].values, "y": y0})
         .groupby("業界").agg(n=("y", "size"), rate=("y", "mean")))
    g = g[g.n >= 10].join(pd.Series(DESK_SHARE, name="prior"))
    print(f"\n  業界レベル(n>=10, {len(g)}業界) デスク比率priorと購入率: "
          f"spearman {spearmanr(g['prior'], g['rate']).statistic:+.3f}")
    print("  ＝ 『机の前にいる従業員がいる業種ほど買う』は業界単位では成り立つ。")

    print("\n=== 4. その構造情報は、既に合成pに入っているか（残差AUC、バー0.55）===")
    rows = []
    for seed in seeds:
        y, p, cache = _blend_and_text(seed, config)
        e3l = _logit(cache[f"oof_{TEXT_EXPERT}"])
        S = struct.values
        cands = {
            "構造5列（拠点・比率）": S[:, :5],
            "デスク比率prior単独": S[:, 5:6],
            "構造+prior": S,
            "熱量×構造（交互作用）": np.column_stack([S, S * e3l[:, None], e3l]),
        }
        for name, X in cands.items():
            o = _crossfit(X, y, seed)
            rows.append({"cand": name, "solo": roc_auc_score(y, o),
                         "corr": spearmanr(_logit(o), _logit(p)).statistic,
                         "resid": _resid_auc(o, p, y)})
    r = pd.DataFrame(rows).groupby("cand")[["solo", "corr", "resid"]].mean()
    print(f"{'候補':<24}{'単体AUC':>9}{'合成と相関':>11}{'残差AUC':>9}")
    for name, row in r.iterrows():
        print(f"{name:<24}{row.solo:>9.3f}{row['corr']:>11.3f}{row.resid:>9.3f}")
    print("\n  残差AUC 0.55 が損益分岐（h92 9.4）。")


# ---------------------------------------------------------------------------
# mode: content -- 商材の中身をメニューから当てる
# ---------------------------------------------------------------------------
def _encode_texts(texts, anchors):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E5_MODEL, local_files_only=True)
    mdl = AutoModel.from_pretrained(E5_MODEL, local_files_only=True).eval()
    torch.set_num_threads(8)

    @torch.no_grad()
    def enc(items, prefix, batch=32):
        out = []
        for i in range(0, len(items), batch):
            e = tok([prefix + str(t) for t in items[i:i + batch]], padding=True,
                    truncation=True, max_length=512, return_tensors="pt")
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
        return np.vstack(out)

    return enc(texts, "passage: ") @ enc(anchors, "query: ").T


def _content_sims(force=False):
    """全文と教育文それぞれについて、12メニューとのcos類似度。train+test。"""
    if os.path.exists(SIM_CACHE) and not force:
        z = np.load(SIM_CACHE, allow_pickle=True)
        return z["full"], z["edu"], int(z["n_train"])
    raw, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    full = pd.concat([raw[TEXT_COL].fillna(""),
                      test[TEXT_COL].fillna("")]).astype(str).values
    edu = np.array([education_context(t) for t in full])
    anchors = [v[0] for v in CONTENTS.values()]
    S_full, S_edu = _encode_texts(full, anchors), _encode_texts(edu, anchors)
    np.savez(SIM_CACHE, full=S_full, edu=S_edu, n_train=len(raw))
    print(f"saved: {SIM_CACHE}")
    return S_full, S_edu, len(raw)


def _maturity_axis(y):
    """DX成熟度軸 = OpenAI埋め込みのPC1。ラベル不使用（dx_text_is_one_dimension）。"""
    z = np.load("data/_emb_dx_outlook_text-embedding-3-large.npz", allow_pickle=True)
    E = np.vstack([z["train"], z["test"]]).astype(np.float64)
    from sklearn.decomposition import PCA
    v = PCA(n_components=1, random_state=0).fit_transform(E)
    if roc_auc_score(y, v[:len(z["train"]), 0]) < 0.5:
        v = -v
    return v


def _row_center(S):
    """行内で中心化して列方向にz化。『どれだけ熱心か』を落とし『何を欲しいか』だけ残す。"""
    C = S - S.mean(axis=1, keepdims=True)
    return (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-12)


def run_content(seeds, config, n_perm, force):
    raw, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    S_full, S_edu, ntr = _content_sims(force)
    names = list(CONTENTS)
    texts_all = pd.concat([raw[TEXT_COL].fillna(""),
                           test[TEXT_COL].fillna("")]).astype(str).values

    print("=== 1. 計器の検証: この類似度は本当に『その話題』を見ているか ===")
    print("  （その中身のキーワードを含む行を、類似度だけで当てられるか）\n")
    print(f"{'メニュー':<20}{'出現率':>8}{'素の類似度':>11}{'行内中心化後':>13}")
    Craw = _row_center(S_full)
    ok = 0
    for i, n in enumerate(names):
        flag = pd.Series(texts_all).str.contains(CONTENTS[n][1], regex=True).values
        a1 = roc_auc_score(flag, S_full[:, i])
        a2 = roc_auc_score(flag, Craw[:, i])
        ok += a2 >= 0.60
        print(f"{n:<20}{flag.mean():>8.3f}{a1:>11.3f}{a2:>13.3f}")
    print(f"\n  行内中心化後に AUC>=0.60 を満たすメニュー: {ok}/{len(names)}"
          "  ← ここが通らなければ以降の帰無は語彙のせいになる")

    y = raw["購入フラグ"].values
    Ctr, Ctr_edu = Craw[:ntr], _row_center(S_edu)[:ntr]
    pc1 = _maturity_axis(y)[:ntr]

    print("\n=== 2. 商材の中身をラベルに選ばせる（全742行、行内中心化＝熱量は消してある）===")
    print("  最後の列は、DX成熟度(PC1)を線形に抜いた後にまだ残る向き。"
          "\n  ここが 0.5 に戻る中身は『範囲』ではなく熱量の言い換えだったということ。\n")
    print(f"{'メニュー':<20}{'全文AUC':>9}{'  95%CI':>17}{'教育文AUC':>11}"
          f"{'PC1除去後':>10}{'  買う側?':>9}")
    rng = np.random.default_rng(0)
    tab = []
    for i, n in enumerate(names):
        a = roc_auc_score(y, Ctr[:, i])
        bs = [roc_auc_score(y[j], Ctr[j, i]) for j in
              (rng.choice(len(y), len(y), True) for _ in range(1000))]
        lo, hi = np.quantile(bs, [.025, .975])
        ae = roc_auc_score(y, Ctr_edu[:, i])
        res = Ctr[:, i] - LinearRegression().fit(pc1, Ctr[:, i]).predict(pc1)
        ar = roc_auc_score(y, res)
        sig = "◯" if lo > 0.5 else ("×（買わない側）" if hi < 0.5 else "")
        tab.append((n, a, lo, hi, ae, ar))
        print(f"{n:<20}{a:>9.3f}  [{lo:.3f},{hi:.3f}]{ae:>11.3f}"
              f"{ar:>10.3f}{sig:>9}")
    tab.sort(key=lambda t: -t[1])
    print(f"\n  → 購入側に最も寄る中身: {tab[0][0]} (AUC {tab[0][1]:.3f}"
          f"、PC1除去後 {tab[0][5]:.3f}) / "
          f"最も買わない側: {tab[-1][0]} (AUC {tab[-1][1]:.3f}"
          f"、PC1除去後 {tab[-1][5]:.3f})")

    print("\n=== 3. 仮説の直接検定 — 熱心なのに買わない企業は『違う中身』を求めているか ===")
    print("  熱心群のなかで、行内中心化した12次元だけで 買う/買わない を分ける\n")
    obs, nulls = [], []
    for seed in seeds:
        yy, p, cache = _blend_and_text(seed, config)
        hot = _hot_mask(cache[f"oof_{TEXT_EXPERT}"])
        sub = np.where(hot)[0]
        o = _crossfit(Ctr[sub], yy[sub], seed)
        obs.append(roc_auc_score(yy[sub], o))
    obs_m = float(np.mean(obs))
    yy, p, cache = _blend_and_text(seeds[-1], config)
    hot = _hot_mask(cache[f"oof_{TEXT_EXPERT}"])
    sub = np.where(hot)[0]
    for _ in range(n_perm):
        yp = rng.permutation(yy[sub])
        nulls.append(roc_auc_score(yp, _crossfit(Ctr[sub], yp, 0)))
    q95 = float(np.quantile(nulls, 0.95))
    pval = (np.sum(np.array(nulls) >= obs_m) + 1) / (len(nulls) + 1)
    print(f"  観測 CV-AUC {obs_m:.3f} (11seed平均, 各seed {np.min(obs):.3f}-{np.max(obs):.3f})")
    print(f"  ラベル入替{n_perm}回の帰無: 平均 {np.mean(nulls):.3f} "
          f"sd {np.std(nulls):.3f} 95%点 {q95:.3f}   経験的p {pval:.3f}")
    print(f"  判定: {'帰無を割る' if obs_m > q95 else '帰無を割らない＝中身の違いでは説明されない'}")

    print("\n  メニュー別の 熱心・買う vs 熱心・買わない の差（+なら買う側に多い中身）")
    hb, hn = hot & (yy == 1), hot & (yy == 0)
    for i, n in enumerate(names):
        d = Ctr[hb, i].mean() - Ctr[hn, i].mean()
        se = np.sqrt(Ctr[hb, i].var() / hb.sum() + Ctr[hn, i].var() / hn.sum())
        mark = "  ← 0を跨がない" if abs(d) > 1.96 * se else ""
        print(f"    {n:<20}{d:>+7.3f}  [{d - 1.96 * se:+.3f},{d + 1.96 * se:+.3f}]{mark}")

    print("\n=== 4. 12次元は既存の合成に何か足すか（残差AUC、バー0.55）===")
    rows = []
    for seed in seeds:
        yy, p, cache = _blend_and_text(seed, config)
        for nm, X in (("中身12次元(全文)", Ctr), ("中身12次元(教育文)", Ctr_edu),
                      ("中身12次元+素の類似度", np.hstack([Ctr, S_full[:ntr]]))):
            o = _crossfit(X, yy, seed)
            rows.append({"cand": nm, "solo": roc_auc_score(yy, o),
                         "corr": spearmanr(_logit(o), _logit(p)).statistic,
                         "resid": _resid_auc(o, p, yy)})
    r = pd.DataFrame(rows).groupby("cand")[["solo", "corr", "resid"]].mean()
    print(f"{'候補':<24}{'単体AUC':>9}{'合成と相関':>11}{'残差AUC':>9}")
    for name, row in r.iterrows():
        print(f"{name:<24}{row.solo:>9.3f}{row['corr']:>11.3f}{row.resid:>9.3f}")


# ---------------------------------------------------------------------------
# mode: pocket -- 「範囲外のポケット」の全探索
# ---------------------------------------------------------------------------
def _condition_bank(raw, sims=None):
    """ラベル非依存の二値条件を全部並べる。(名前, bool配列) のリスト。"""
    C = []
    for c in CAT_COLS:
        for lv, cnt in raw[c].value_counts().items():
            if cnt >= 8:
                C.append((f"{c}={lv}", (raw[c] == lv).values))
    num = raw.select_dtypes(include=[np.number]).drop(columns=["企業ID", "購入フラグ"])
    for c in num.columns:
        v = num[c].values.astype(float)
        if np.isnan(v).any():
            C.append((f"{c}=欠損", np.isnan(v)))
        q = np.nanquantile(v, [.25, .5, .75])
        for lab, cond in ((f"{c}<=Q1", v <= q[0]), (f"{c}<=中央", v <= q[1]),
                          (f"{c}>=中央", v >= q[1]), (f"{c}>=Q3", v >= q[2])):
            C.append((lab, np.nan_to_num(cond, nan=False).astype(bool)))
    F = build_material_scope_features(raw)
    for k in [*IN_SCOPE, *OUT_SCOPE]:
        C.append((f"文:{k}", F[k].astype(bool)))
    S = build_structure_features(raw)
    for c in STRUCT_COLS:
        v = S[c].values
        med = np.median(v)
        C.append((f"構造:{c}<=中央", v <= med))
        C.append((f"構造:{c}>=中央", v >= med))
    if sims is not None:
        for i, n in enumerate(CONTENTS):
            v = sims[:, i]
            t1, t2 = np.quantile(v, [1 / 3, 2 / 3])
            C.append((f"中身:{n}下位1/3", v <= t1))
            C.append((f"中身:{n}上位1/3", v >= t2))
    return C


def _search(M, y0, min_support):
    """単条件とAND2条件を全部評価し、非購入精度の最大を返す。

    M: (n_cond, n_rows) bool。y0: 非購入なら1。
    """
    A = M.astype(np.float32)
    sup1 = A.sum(1)
    hit1 = A @ y0
    best = (0.0, 0, "")
    with np.errstate(invalid="ignore", divide="ignore"):
        prec1 = np.where(sup1 >= min_support, hit1 / np.maximum(sup1, 1), -1)
    i = int(np.argmax(prec1))
    if prec1[i] > best[0]:
        best = (float(prec1[i]), int(sup1[i]), str(i))
    sup2 = A @ A.T
    hit2 = (A * y0) @ A.T
    with np.errstate(invalid="ignore", divide="ignore"):
        prec2 = np.where(sup2 >= min_support, hit2 / np.maximum(sup2, 1), -1)
    np.fill_diagonal(prec2, -1)
    j = int(np.argmax(prec2))
    a, b = divmod(j, prec2.shape[1])
    if prec2[a, b] > best[0]:
        best = (float(prec2[a, b]), int(sup2[a, b]), f"{a}&{b}")
    return best


def run_pocket(seeds, config, n_perm, min_support):
    raw = pd.read_csv("data/train.csv")
    sims = None
    if os.path.exists(SIM_CACHE):
        z = np.load(SIM_CACHE, allow_pickle=True)
        sims = _row_center(z["full"])[:int(z["n_train"])]
        print(f"（{SIM_CACHE} を読めたので中身12メニューも条件に含める）")
    bank = _condition_bank(raw, sims)
    names = [n for n, _ in bank]
    B = np.vstack([v for _, v in bank])
    print(f"条件バンク: 単条件 {len(bank)}本、AND2まで含めて "
          f"{len(bank) * (len(bank) - 1) // 2 + len(bank):,}通りを全探索\n")

    y0, p0, cache0 = _blend_and_text(seeds[-1], config)
    f1_now = max(f1_score(y0, (p0 >= t).astype(int)) for t in THS)
    bar = 1 - f1_now / 2
    print(f"現状F1 {f1_now:.4f} → 拒否ルールの損益分岐は精度 1-F1/2 = {bar:.3f}"
          f"（h92 9.2）\n")

    rng = np.random.default_rng(0)
    for label, getter in (("全742行（『範囲外なら誰も買わない』の最強形）",
                           lambda y, p, c: np.ones(len(y), bool)),
                          ("高p行（p>0.5、FPが住む所）",
                           lambda y, p, c: p > 0.5),
                          (f"テキスト熱心群（{TEXT_EXPERT}上位{HOT_FRAC:.0%}）",
                           lambda y, p, c: _hot_mask(c[f"oof_{TEXT_EXPERT}"]))):
        print(f"--- 条件付け集合: {label} ---")
        obs, sup, rule, nulls = [], [], None, []
        y0_, p0_, c0_ = _blend_and_text(seeds[-1], config)
        # 支持は群サイズに比例させる。非購入率が高い群では小さいポケットは
        # 入替でも簡単に純粋になり、検定の検出力がゼロになるため。
        ms = max(min_support, int(round(0.08 * getter(y0_, p0_, c0_).sum())))
        for seed in seeds:
            y, p, cache = _blend_and_text(seed, config)
            m = getter(y, p, cache)
            M = B[:, m]
            prec, s, idx = _search(M, (y[m] == 0).astype(np.float32), ms)
            obs.append(prec)
            sup.append(s)
            if seed == seeds[-1]:
                rule = " AND ".join(names[int(t)] for t in idx.split("&"))
        y, p, cache = _blend_and_text(seeds[-1], config)
        m = getter(y, p, cache)
        M = B[:, m]
        for _ in range(n_perm):
            yp = rng.permutation((y[m] == 0).astype(np.float32))
            nulls.append(_search(M, yp, ms)[0])
        q95 = float(np.quantile(nulls, 0.95))
        obs_m = float(np.mean(obs))
        pv = (np.sum(np.array(nulls) >= obs_m) + 1) / (len(nulls) + 1)
        print(f"  対象 {int(m.sum())}行（非購入 {int((y[m] == 0).sum())}行 = "
              f"{(y[m] == 0).mean():.1%}）  最小支持 {ms}")
        print(f"  最良の拒否ルール（seed{seeds[-1]}）: {rule}")
        print(f"    非購入精度 {obs[-1]:.3f}  支持 {sup[-1]}行 "
              f"／ 11seed平均 精度 {obs_m:.3f}")
        print(f"  同じ全探索をラベル入替{n_perm}回: 平均 {np.mean(nulls):.3f} "
              f"95%点 {q95:.3f}  経験的p {pv:.3f}")
        verdict_txt = ("帰無を割る" if obs_m > q95 else
                       "帰無を割らない＝探索の多重性で説明できる")
        print(f"  判定: {verdict_txt}／損益分岐 {bar:.3f} に対し "
              f"{'到達' if obs_m >= bar else '未達'}\n")
        if obs_m > q95:
            _validate_pocket(B, names, getter, seeds, config, ms, n_perm, rng)


def _apply(B, idx):
    parts = [int(t) for t in idx.split("&")]
    m = B[parts[0]].copy()
    for t in parts[1:]:
        m &= B[t]
    return m


def _validate_pocket(B, names, getter, seeds, config, min_support, n_perm, rng):
    """見つけたポケットを (1) 現行モデルは既に知っているか (2) fold外へ出るか
    (3) F1が動くか、の3点で確かめる。探索は必ずfoldのtrain側だけで行う。"""
    print("  --- 検証 ---")
    y, p, cache = _blend_and_text(seeds[-1], config)
    m = getter(y, p, cache)
    prec, sup, idx = _search(B[:, m], (y[m] == 0).astype(np.float32), min_support)
    hit = _apply(B, idx) & m
    print(f"  (1) 該当{int(hit.sum())}行の現行モデルの予測: "
          f"p中央値 {np.median(p[hit]):.3f}、p>0.5は {int((p[hit] > 0.5).sum())}行"
          f"（＝閾値0.5では既に非購入と言えている行数 {int((p[hit] <= 0.5).sum())}）")

    print("  (2) fold外汎化: ルールは各foldのtrain側だけで探し、val側で精度を測る")
    obs, null = [], []
    for shuffle in (False, True):
        for seed in seeds if not shuffle else seeds[:3]:
            yy, pp, cc = _blend_and_text(seed, config)
            mm = getter(yy, pp, cc)
            yv = yy.copy()
            if shuffle:      # 条件付け集合の中だけで入替（群の購入率を保つ）
                yv[mm] = rng.permutation(yv[mm])
            hits = wins = 0
            for tr, va in StratifiedKFold(
                    5, shuffle=True, random_state=seed).split(B.T, yv):
                tm = np.zeros(len(yv), bool)
                tm[tr] = True
                sel = mm & tm
                if sel.sum() < min_support * 2:
                    continue
                _, _, ix = _search(B[:, sel], (yv[sel] == 0).astype(np.float32),
                                   max(4, min_support // 2))
                vm = np.zeros(len(yv), bool)
                vm[va] = True
                h = _apply(B, ix) & mm & vm
                hits += int(h.sum())
                wins += int((yv[h] == 0).sum())
            (null if shuffle else obs).append(
                (wins / hits if hits else np.nan, hits))
    o = np.array([v for v, _ in obs], float)
    nl = np.array([v for v, _ in null], float)
    print(f"      val側の非購入精度 {np.nanmean(o):.3f}"
          f"（該当 {np.mean([h for _, h in obs]):.1f}行/seed、"
          f"{len(seeds)}seed {np.nanmin(o):.3f}-{np.nanmax(o):.3f}）")
    print(f"      ラベル入替時の同じ手続き {np.nanmean(nl):.3f}"
          f"（3seed）  ＝ 探索は入替でも同じくらい当たるか")

    print("  (3) 2条件の分解（片方だけでは足りないのか）")
    parts = [int(t) for t in idx.split("&")]
    for lab, cond in ([(names[t], B[t]) for t in parts]
                      + [("両方", _apply(B, idx))]):
        s = cond & m
        print(f"      {lab:<34} {int(s.sum()):>4}行  非購入率 "
              f"{(y[s] == 0).mean():.3f}")

    print("  (4) F1: 該当行を0に倒したときの本番foldでの差")
    d, dfix = [], []
    for seed in seeds:
        yy, pp, cc = _blend_and_text(seed, config)
        mm = getter(yy, pp, cc)
        _, _, ix = _search(B[:, mm], (yy[mm] == 0).astype(np.float32), min_support)
        h = _apply(B, ix) & mm
        q = pp.copy()
        q[h] = 0.0
        d.append(max(f1_score(yy, (q >= t).astype(int)) for t in THS)
                 - max(f1_score(yy, (pp >= t).astype(int)) for t in THS))
        dfix.append(f1_score(yy, (q >= 0.5).astype(int))
                    - f1_score(yy, (pp >= 0.5).astype(int)))
    d, dfix = np.array(d), np.array(dfix)
    print(f"      OOF最適閾値 ΔF1 {d.mean():+.4f}（正 {int((d > 0).sum())}/{len(d)}）"
          "  ← 閾値のargmaxが動くので過大に出る")
    print(f"      固定閾値0.5  ΔF1 {dfix.mean():+.4f}"
          f"（正 {int((dfix > 0).sum())}/{len(dfix)}）  ← こちらが実力"
          "\n      ※どちらもルールは評価と同じ行で選んでいるので上界\n")


# ---------------------------------------------------------------------------
# mode: screen -- 本番の判定装置（11seed、9本目のExpert）
# ---------------------------------------------------------------------------
def run_screen(seeds, config):
    raw, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    Str, Ste = build_structure_features(raw).values, build_structure_features(test).values
    cands = {"構造(デスク比率)": (Str, Ste)}
    if os.path.exists(SIM_CACHE):
        z = np.load(SIM_CACHE, allow_pickle=True)
        ntr = int(z["n_train"])
        C = _row_center(z["full"])
        cands["中身12メニュー"] = (C[:ntr], C[ntr:])
        cands["構造+中身"] = (np.hstack([Str, C[:ntr]]), np.hstack([Ste, C[ntr:]]))
    else:
        print(f"（{SIM_CACHE} が無いので構造のみ。--mode content を先に実行）")

    rows = []
    for seed in seeds:
        cache = np.load(CONFIGS[config].format(seed=seed), allow_pickle=True)
        y = cache["y"]
        oof = {n: cache["oof_" + n] for n in EXPERTS}
        te = {n: cache["te_" + n] for n in EXPERTS}
        base, _, _, _ = blend_oof(y, oof, te, seed=seed, experts=EXPERTS)
        rows.append({"seed": seed, "arm": "base", **_metrics(y, base)})
        e3l = _logit(cache[f"oof_{TEXT_EXPERT}"])
        te3l = _logit(cache[f"te_{TEXT_EXPERT}"])
        arms = dict(cands)
        arms["熱量×構造"] = (np.column_stack([Str, Str * e3l[:, None], e3l]),
                             np.column_stack([Ste, Ste * te3l[:, None], te3l]))
        # 対照。E3のOOFをそのまま9本目にするだけ。交互作用の効果を主張するには、
        # この対照を超えなければならない（メタでE3に重みを二重に配るのと同じ）。
        arms["対照:熱量だけ再投入"] = (e3l.reshape(-1, 1), te3l.reshape(-1, 1))
        for name, (A, Bt) in arms.items():
            o, t = _crossfit(A, y, seed, Xte=Bt)
            cand, _, _, _ = blend_oof(y, {**oof, "E8": o}, {**te, "E8": t},
                                      seed=seed, experts=list(EXPERTS) + ["E8"])
            rows.append({"seed": seed, "arm": name, "solo": roc_auc_score(y, o),
                         **_metrics(y, cand)})
    r = pd.DataFrame(rows)
    b = r[r.arm == "base"].sort_values("seed").reset_index(drop=True)
    print(f"\n=== 9本目として本番の外側foldに足す（{len(seeds)}seed）===\n")
    print(f"{'候補':<20}{'単体AUC':>9}{'ΔAUC':>10}{'ΔAP':>10}{'ΔF1':>10}"
          f"{'ΔAP>0':>8}  判定")
    for name in [a for a in r.arm.unique() if a != "base"]:
        c = r[r.arm == name].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{name:<20}{c.solo.mean():>9.3f}{(c.auc - b.auc).mean():>+10.4f}"
              f"{(c.ap - b.ap).mean():>+10.4f}{(c.f1 - b.f1).mean():>+10.4f}"
              f"{int((c.ap > b.ap).sum()):>6}/{len(c)}  {v}")
    r.to_csv("exp/_h96_screen.csv", index=False)
    print("\nsaved: exp/_h96_screen.csv")


def _metrics(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": max(f1_score(y, (p >= t).astype(int)) for t in THS),
            "f1_50": f1_score(y, (p >= 0.5).astype(int))}


# ---------------------------------------------------------------------------
# mode: gate -- ポケットを本番の外側foldの中だけで探して当てる（strict）
# ---------------------------------------------------------------------------
def run_gate(seeds, config, min_support):
    """pocket が見つけたルールを、fold の train 側だけで探し直して val に当てる。

    ルールの探索・支持・精度の判断が全部 fold の中で閉じるので、これは
    「本番で運用できるか」の判定になる。fold が条件を満たすルールを
    見つけられなければゲートを開かない（＝何もしない）ので、
    ルールが実在しないなら結果はベースラインと一致するはずである。
    """
    raw = pd.read_csv("data/train.csv")
    sims = None
    if os.path.exists(SIM_CACHE):
        z = np.load(SIM_CACHE, allow_pickle=True)
        sims = _row_center(z["full"])[:int(z["n_train"])]
    bank = _condition_bank(raw, sims)
    names = [n for n, _ in bank]
    B = np.vstack([v for _, v in bank])

    rows, audit, picked = [], np.zeros(4), {}
    for seed in seeds:
        y, p, _ = _blend_and_text(seed, config)
        base = _metrics(y, p)
        rows.append({"seed": seed, "arm": "base", "gated": 0, **base})
        bar = 1 - base["f1"] / 2
        for arm in ("gate_hard", "gate_guarded"):
            q, n_gate = p.copy(), 0
            for tr, va in StratifiedKFold(
                    5, shuffle=True, random_state=seed).split(B.T, y):
                prec, sup, idx = _search(B[:, tr], (y[tr] == 0).astype(np.float32),
                                         min_support)
                if arm == "gate_guarded" and prec < bar:
                    continue        # fold が自分でゲートを開かないと決められる
                hit = np.zeros(len(y), bool)
                hit[va] = True
                hit &= _apply(B, idx)
                q[hit] = 0.0
                n_gate += int(hit.sum())
                if arm == "gate_hard":
                    rule = " AND ".join(names[int(t)] for t in idx.split("&"))
                    picked[rule] = picked.get(rule, 0) + 1
                    audit += [hit.sum(), (y[hit] == 1).sum(),
                              (hit & (p > 0.5)).sum(),
                              (hit & (p > 0.5) & (y == 1)).sum()]
            rows.append({"seed": seed, "arm": arm, "gated": n_gate,
                         **_metrics(y, q)})
    audit /= len(seeds)
    r = pd.DataFrame(rows)
    b = r[r.arm == "base"].sort_values("seed").reset_index(drop=True)
    print("=== ポケットをfoldの中だけで探して当てる（11seed、本番外側fold）===\n")
    print(f"{'arm':<16}{'倒した行/seed':>13}{'ΔAUC':>10}{'ΔAP':>10}"
          f"{'ΔF1(最適)':>11}{'ΔF1(0.5)':>10}{'ΔAP>0':>8}  判定")
    for arm in ("gate_hard", "gate_guarded"):
        c = r[r.arm == arm].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{arm:<16}{c.gated.mean():>13.1f}{(c.auc - b.auc).mean():>+10.4f}"
              f"{(c.ap - b.ap).mean():>+10.4f}{(c.f1 - b.f1).mean():>+11.4f}"
              f"{(c.f1_50 - b.f1_50).mean():>+10.4f}"
              f"{int((c.ap > b.ap).sum()):>6}/{len(c)}  {v}")
    print("\n--- なぜ精度0.94のルールが負けるのか（1seedあたり、gate_hard）---")
    print(f"  倒した行 {audit[0]:.1f}／うち実は購入者 {audit[1]:.1f}")
    print(f"  そのうち p>0.5 の行は {audit[2]:.1f} 行しかない"
          f"（うちTP {audit[3]:.1f} を壊し、FP {audit[2] - audit[3]:.1f} を直す）")
    print(f"  ＝ 効く精度は {1 - audit[3] / max(audit[2], 1e-9):.3f}。"
          "\n  拒否ルールの損得は『倒した行の非購入率』ではなく"
          "\n  『倒した p>0.5 行の非購入率』で決まる（h92 9.2 の基準の訂正）。"
          "\n  モデルが既に落としている行を重ねて落としても、F1は1ミリも動かない。")
    print("\n  foldが選んだルール（上位）")
    for rule, cnt in sorted(picked.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {cnt:>3}/{len(seeds) * 5}fold  {rule}")
    r.to_csv("exp/_h96_gate.csv", index=False)
    print("\nsaved: exp/_h96_gate.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="premise",
                    choices=("premise", "content", "pocket", "screen", "gate"))
    ap.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--min-support", type=int, default=12)
    ap.add_argument("--force-encode", action="store_true")
    args = ap.parse_args()
    if args.mode == "premise":
        run_premise(SEEDS, args.config)
    elif args.mode == "content":
        run_content(SEEDS, args.config, min(args.n_perm, 60), args.force_encode)
    elif args.mode == "pocket":
        run_pocket(SEEDS, args.config, args.n_perm, args.min_support)
    elif args.mode == "gate":
        run_gate(SEEDS, args.config, args.min_support)
    else:
        run_screen(SEEDS, args.config)


if __name__ == "__main__":
    main()
