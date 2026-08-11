"""H22: アンケート11項目のドメイン派生特徴（E2 アンケートエキスパート専用）。

⚠️ **結論: exp025 で REJECT**。既定は無効（`e2_survey_cols(with_step=False)`）。
   `step` 群は E2単体では 20seedペア比較で ΔAP +0.0151 (20/20) の ACCEPT(明確) だが、
   **合成後のホールドアウトで ΔAP -0.0051 (1/5) / ΔF1 -0.0030 (2/5)** とガードレール違反。
   原因は「単体の改善を直交性で買っていた」こと: step の4本は a1/a2/a4/a10 を閾値で
   切った列で、生値は E0 の94列に入っており **E0 の木が同じ分割を自分で作れる**。
   結果 E2 と他エキスパートの相関が全面的に上がり（E6 +0.042 / E3 +0.029 / E0 +0.026）、
   メタが E2 の重みを 32.8% → 29.1% に下げた。詳細は `documents/experiments.md` の exp025。
   コードは再検証用に残す（`--survey-step`）。

動機（当時の想定。結果としてこの読み自体は当たっていて、外したのは採り方）:
  E2 は合成メタで **最大の重み**（holdout rep平均 33.0%）を取り、E0(94列)との
  相関は 0.187 しかない ＝ 単一モデルが取りこぼしている領域を担当している。
  さらに E2 は診断で **LR が LGBM に AP +0.048 で明確勝ち**している
  （742行では木が1〜5の順序尺度11本を刻みすぎる）。
  つまりこのグループの非線形性は、木に学習させるのではなく
  **手で列にして LR に渡すしかない**。E1(財務)と違い改善の伝達率も高い。

候補の選び方 — LR が原理的に作れないものだけを入れる:
  - 差分 `a1 - a2` や重み付き和は LR が自前で作れる（w1*a1 + w2*a2 と同型）。
    「ギャップ特徴」として差を足すのは **無意味なので入れない**。
  - 作れないのは 積・順序統計量(std/min/max/range)・閾値指標(a>=4)・
    欠損条件付きの合成 の4種類。ここだけを候補にする。

事前診断（train 742行 / 正例率 0.2412 / AP倍率＝単体AP÷正例率 /
最大|r| は既存12列との相関の最大値）:

  群      列                AUC    AP    AP倍  最大|r|
  gap    アンケ_ツール不満   0.606  0.317  1.31  0.688
  gap    アンケ_満足x成果    0.573  0.283  1.18  0.658
  gap    アンケ_情報x連携    0.567  0.281  1.17  0.696
  gap    アンケ_明確x連携    0.543  0.274  1.14  0.690
  shape  アンケ_分散9        0.525  0.277  1.15  **0.076**
  shape  アンケ_中庸数       0.510  0.265  1.10  **0.079**
  shape  アンケ_中央偏差和   0.503  0.257  1.07  **0.116**
  shape  アンケ_レンジ9      0.511  0.256  1.06  **0.048**
  step   アンケ_連携高       0.583  0.278  1.15  0.861
  step   アンケ_抵抗高       0.578  0.275  1.14  0.878
  step   アンケ_明確高       0.553  0.265  1.10  0.868
  step   アンケ_満足低       0.527  0.252  1.04  0.874

⚠️ **この事前診断はスクリーニングとして機能しなかった**（exp025 の副産物）。
  実測（E2単体20seedペア比較）とランキングが逆になった:
    gap   単体AP最上位(1.14〜1.31) / |r| 0.66〜0.70  → **0/20 で悪化 REJECT**
    step  単体AP中位(1.04〜1.15)   / |r| 0.86〜0.88  → **20/20 で ACCEPT(明確)**
    shape 単体AP中位              / |r| 0.05〜0.12  → ノイズ
  積は単体APが高くても線形項との重複で新規情報がなく、閾値指標は生値と|r|=0.87 でも
  線形空間に無い成分（折れ線）を持つ。H21(木向け)の足切り手順は LR には流用できない。
  **判定は必ず `compare_survey.py` の同一seedペア比較で付けること**（E2はLRなので数秒）。
  却下則1のとおり、直交性(shape)も採用根拠にはならなかった。

構造メモ:
  - `アンケート７`(既存ツール満足度) は `アンケート６`＝いいえ の356行(48%)で欠損。
    欠損は完全に a6==2 と一致するので、`アンケート７_isna` は a6 と**完全共線**。
  - a7 は単体で最強（購入率 1:0.464 → 5:0.127, 相関 -0.258）。
  - `harness.preprocess` の ZERO_FILL が a7 を 0 で埋めるため、標準化前の分布が
    0 と 1〜5 の二峰になる。欠損行は isna フラグで独自の切片を持てるので表現力は
    落ちないが、σ が膨らんで観測値1〜5の z 幅が縮み、**最強項目が L2 で過剰に
    縮む**。`a7_median_fill` はこれを外側train内の中央値埋めに置き換える変種。

行単位の決定的変換のみ（a7中央値埋めだけは fold内 train 統計を使う）＝リーク無し。
"""
import numpy as np
import pandas as pd

# 常に観測される5段階9項目。a6は2値、a7は条件付き欠損なので分布統計から外す。
FIVE_POINT_COLS = (
    "アンケート１", "アンケート２", "アンケート３", "アンケート４", "アンケート５",
    "アンケート８", "アンケート９", "アンケート１０", "アンケート１１",
)
BINARY_COL = "アンケート６"
COND_COL = "アンケート７"
COND_ISNA_COL = "アンケート７_isna"

SHAPE_COLS = ("アンケ_分散9", "アンケ_レンジ9", "アンケ_中庸数", "アンケ_中央偏差和")
STEP_COLS = ("アンケ_明確高", "アンケ_満足低", "アンケ_抵抗高", "アンケ_連携高")
GAP_COLS = ("アンケ_ツール不満", "アンケ_満足x成果", "アンケ_情報x連携", "アンケ_明確x連携")

SURVEY_GROUPS = {"shape": SHAPE_COLS, "step": STEP_COLS, "gap": GAP_COLS}
ALL_DERIVED_COLS = SHAPE_COLS + STEP_COLS + GAP_COLS

# a6==いいえ(未導入)を a7 スケールのどこに置くか。未導入の購入率 0.222 は
# a7=3(0.247) と a7=4(0.169) の間なので 3.3。不満度は 6-a7 で向きを揃える。
_UNINSTALLED_A7 = 3.3


def _five(raw):
    return raw[list(FIVE_POINT_COLS)].astype(float)


def add_survey_features(raw, X, groups=("shape", "step", "gap")):
    """`raw`(生train/test) から派生列を作り、`X` のコピーに足して返す。

    既存の add_* と同じ (raw, X) -> X 規約。X は破壊しない。
    `groups` で群を絞れる（既定は全部）。未知の群名は例外にする。
    """
    unknown = [g for g in groups if g not in SURVEY_GROUPS]
    if unknown:
        raise ValueError(f"未知の群: {unknown} (有効: {sorted(SURVEY_GROUPS)})")

    F = _five(raw)
    a = {int(n): raw[c].astype(float)
         for n, c in zip([1, 2, 3, 4, 5, 8, 9, 10, 11], FIVE_POINT_COLS)}
    a[6], a[7] = raw[BINARY_COL].astype(float), raw[COND_COL].astype(float)

    new = {}
    if "shape" in groups:
        # 順序統計量。LRは行内のばらつきを列の線形結合では作れない。
        new["アンケ_分散9"] = F.std(axis=1)
        new["アンケ_レンジ9"] = F.max(axis=1) - F.min(axis=1)
        new["アンケ_中庸数"] = (F == 3).sum(axis=1).astype(float)
        new["アンケ_中央偏差和"] = (F - 3).abs().sum(axis=1)
    if "step" in groups:
        # 購入率カーブが単調でない項目の閾値化。線形項と併用で折れ線になる。
        new["アンケ_明確高"] = (a[1] >= 4).astype(float)
        new["アンケ_満足低"] = (a[2] <= 2).astype(float)
        new["アンケ_抵抗高"] = (a[4] >= 4).astype(float)
        new["アンケ_連携高"] = (a[10] >= 4).astype(float)
    if "gap" in groups:
        # 積。「意欲が高いときだけ不満が効く」型の条件付き効果を表す。
        # 差分ではなく積にするのは、差分をLRが自前で作れるため。
        new["アンケ_ツール不満"] = np.where(a[6] == 2, _UNINSTALLED_A7,
                                       6 - a[7].fillna(3))
        new["アンケ_満足x成果"] = a[2] * a[8]
        new["アンケ_情報x連携"] = a[11] * a[10]
        new["アンケ_明確x連携"] = a[1] * a[10]

    out = X.copy()
    for c, v in new.items():
        out[c] = np.asarray(v, dtype=float)
    return out


def a7_median_fill(Xtr, *others):
    """a7 の 0埋め(欠損マーカー)を、**Xtr の観測中央値**に置き換えた組を返す。

    欠損行は `アンケート７_isna == 1` で識別する（a7∈1..5 なので 0 は必ず欠損）。
    中央値は Xtr からのみ取るので、valid/test 側にリークしない。
    a7列 or isna列が無い場合はそのまま返す（列を絞った実験でも壊さない）。
    """
    if COND_COL not in Xtr.columns or COND_ISNA_COL not in Xtr.columns:
        return (Xtr,) + tuple(others)
    obs = Xtr.loc[Xtr[COND_ISNA_COL] == 0, COND_COL]
    fill = float(obs.median()) if len(obs) else 0.0

    def _apply(df):
        out = df.copy()
        out.loc[out[COND_ISNA_COL] == 1, COND_COL] = fill
        return out

    return tuple(_apply(d) for d in (Xtr,) + tuple(others))


def survey_feature_report(raw, y):
    """事前診断用: 各派生列の単体AUC/AP と既存12列との最大|r| を返す。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    base = raw[list(FIVE_POINT_COLS) + [BINARY_COL]].astype(float).copy()
    base[COND_COL] = raw[COND_COL].astype(float).fillna(0)
    base[COND_ISNA_COL] = raw[COND_COL].isna().astype(float)
    X = add_survey_features(raw, base[[]].copy())

    rows = []
    for g, cols in SURVEY_GROUPS.items():
        for c in cols:
            v = X[c].values
            auc, ap = roc_auc_score(y, v), average_precision_score(y, v)
            rows.append(dict(
                群=g, 列=c, AUC=max(auc, 1 - auc),
                AP=max(ap, average_precision_score(y, -v)),
                AP倍=max(ap, average_precision_score(y, -v)) / y.mean(),
                最大r=base.apply(lambda s: abs(np.corrcoef(s, v)[0, 1])).max(),
            ))
    return pd.DataFrame(rows).sort_values("AP", ascending=False)
