"""H27: 「不満スコア × 財務」のグループ横断交互作用（E7 専用）。

事後訂正（2026-08-15 / H34後のアブレーション）:
  E7の成功を「積が効いた」と解釈していたが、seed42の同一5fold LRでは
    親7列（不満軸2 + 財務軸5） AUC 0.8744 / AP 0.6853 / F1 0.6578
    全12列（親7 + 積5）          AUC 0.8747 / AP 0.6859 / F1 0.6562
  だった。積の純増はAP +0.0006、F1 -0.0016で、E7の本体は**強い派生親軸を
  低次元の独立LRにまとめたこと**。クラス名・列名は再現性のため維持するが、
  今後H27を「掛け算が効く根拠」として引用しないこと。

出所:
  チームの田中さんの実験ログ（`documents/experiments_with_tanaka.md` の nb006 / nb008）。
  向こうの単一LightGBMライン上では、この交互作用ブロックが
  ΔAP +0.0232 (10/10) / 財務比の拡張でさらに +0.0058 (8/10) と、
  テキストに次ぐ大きさの改善になっている。

なぜこちらの構成に「無い」と言えるか（当時の移植理由）:
  こちらは exp018 でエキスパートを **E1=財務のみ / E2=アンケートのみ** に
  分離した。この積はどちらのエキスパートも**構造的に作れない**。
  両方を持つのは E0(94列) だけだが、exp018 の診断が
    - E0 と E2 の OOF相関は 0.187 しかない
    - E0 の gain 配分でアンケート12列は 14.7%（1列あたりでは E1 の 1/1.05）
  と示しており、E0 もこの領域を取りこぼしている。
  当時は木が列の積を再構築しにくいことを主因と考えた。事後アブレーションでは
  積より、不満集約と財務比率という派生親軸を専用LRへ隔離した効果が主因だった。

exp020 / exp023 / exp025 の3連敗と何が違うか:
  あの3件はいずれも **既存グループの中身を別表現にして単体を強くする** 形で、
  直交性を失って合成で沈んだ（exp025 は E2 の重みが 32.8%→29.1% に下がった）。
  ここは **グループをまたぐ積** を独立した7本目として足す形なので、
  既存6本の入力は1列も変わらない。E0 は `EXPERT_ONLY_COLS` 経由で
  この列群を見ないので、--cross の有無で E0 の予測は完全一致する
  （--org-embed / --survey-step と同じ設計＝rep単位の厳密なペア比較が取れる）。

事前診断（train 742行 / 正例率 0.2412 / AP倍＝単体AP÷正例率 /
最大|r| は E1・E2 が既に持っている生列との Spearman の最大値）:

  列                        AUC     AP    AP倍   最大|r|
  不満4 x ソフト投資_売上比    0.766  0.534  2.21   0.925  ← 全特徴中で最強
  不満4 x 営業CFマージン      0.743  0.490  2.03   0.953
  不満4 x log_ソフト投資      0.729  0.485  2.01   0.681  ← 最も独立
  不満4 x 自己資本比率        0.672  0.485  2.01   0.855
  不満4 x 有利子負債比率      0.645  0.352  1.46   0.986  ← 最弱
  （参考: 軸そのもの 不満4 0.670/0.399, ソフト投資_売上比 0.724/0.408）

  田中さん側の実測 AUC 0.767 に対しこちらで 0.766、4象限も
  3.2% / 28.7% / 18.2% / 47.0%（向こう 3.8 / 25.7 / 15.6 / 45.4）とほぼ一致。
  却下則「相関が低い＝直交、ではない。単体APが正例率をどれだけ上回るか必ず見る」
  に対しては全5本が 1.46〜2.21倍で通る。

列の採り方（事後の取捨選択を避けるため事前に固定）:
  **5本すべてを採る**。最弱の `x 有利子負債比率` も残すのは、
  田中さん nb008 が「交互作用が **マイナス**(-3.1pt)なのに効く。買う理由ではなく
  **買えない理由**を測るブレーキ役で、他4本のアクセルと補い合う」と
  積極的な理由を挙げているため。単体APの順位で足切りする手続きは
  exp025 でスクリーニングとして機能しなかった実績があるので、ここでは使わない。
  軸そのもの（不満2/不満4 と 財務比5本）も入れる。積だけ渡すと
  「不満が高い」のか「投資が大きい」のかをモデルが分離できない。

符号の注意:
  `無形固定資産変動(ソフトウェア関連)` は **train/test の全1542行が負**
  （投資を支出として負で記録する規約）。素直に log1p を取ると定数列になり
  単体AUC 0.500 になるので、`ソフト投資額 = -無形固定資産変動` として符号を戻す。
  H21 の `ソフト投資比率_売上` は生の符号のまま（木なので順序が反転するだけで
  等価）だが、ここでは log を取る都合と可読性のため符号を戻した別列にしてある。

行単位の決定的変換のみ＝リーク無し（fold内統計を一切使わない）。
"""
import numpy as np
import pandas as pd

# 不満軸に使うアンケート。田中さん nb006/nb007 の「4問版」と同じ選び方。
# 単変数の強さ Q7 0.666 > Q10 0.604 > Q4 0.571 > Q2 0.562 で、5位以下(≤0.548)は入れない。
# 4問の相互相関は最大 0.101 でほぼ独立。
FUMAN_COLS_2 = ("アンケート７", "アンケート１０")
FUMAN_COLS_4 = FUMAN_COLS_2 + ("アンケート２", "アンケート４")

# a6==いいえ(未導入)を a7 スケールのどこに置くか。survey_features._UNINSTALLED_A7 と同値。
# 未導入356社の購入率 0.222 は a7=3(0.247) と a7=4(0.169) の間なので 3.3。
# 0埋めだと「最も不満」側に置かれてしまう。
_UNINSTALLED_A7 = 3.3

AXIS_COLS = ("不満2", "不満4")
RATIO_COLS = (
    "ソフト投資_売上比",
    "営業CFマージン",
    "自己資本比率",
    "log_ソフト投資",
    "有利子負債比率",
)
CROSS_COLS = tuple(f"不満4x{c}" for c in RATIO_COLS)
ALL_CROSS_COLS = AXIS_COLS + RATIO_COLS + CROSS_COLS


def _safe_div(a, b):
    """ゼロ除算・無限大を欠損に落とす。LightGBM は欠損を分岐で扱える。"""
    return (a / b).replace([np.inf, -np.inf], np.nan)


def fuman_score(raw, cols=FUMAN_COLS_4):
    """不満スコア = Σ(6 - 回答)。大きいほど「既存に不満／外部に頼れていない」。

    a7 の欠損（a6==いいえ＝未導入、48%）だけ `_UNINSTALLED_A7` で埋める。
    等重みで足すのは田中さん側の実装に合わせるため（重み付けは未検証課題）。
    """
    s = np.zeros(len(raw), dtype=float)
    for c in cols:
        v = raw[c].astype(float)
        if c == "アンケート７":
            v = v.fillna(_UNINSTALLED_A7)
        s += (6.0 - v).values
    return pd.Series(s, index=raw.index)


def _ratios(raw):
    """財務側の5本。`ソフト投資額` は符号を戻した正の投資額。"""
    soft = -raw["無形固定資産変動(ソフトウェア関連)"].astype(float)
    return {
        "ソフト投資_売上比": _safe_div(soft, raw["売上"].astype(float)),
        "営業CFマージン": _safe_div(raw["営業CF"].astype(float),
                                raw["売上"].astype(float)),
        "自己資本比率": _safe_div(raw["自己資本"].astype(float),
                              raw["総資産"].astype(float)),
        "log_ソフト投資": np.log1p(soft.clip(lower=0)),
        "有利子負債比率": _safe_div(
            (raw["短期借入金"] + raw["長期借入金"]).astype(float),
            raw["総資産"].astype(float)),
    }


def add_cross_features(raw, X):
    """`raw`(生train/test) から H27 の12列を作り、`X` のコピーに足して返す。

    既存の add_* と同じ (raw, X) -> X 規約。X は破壊しない。
    E7 だけが使う列なので、`expert_groups.EXPERT_ONLY_COLS` で E0 から除外される。
    """
    out = X.copy()
    f2, f4 = fuman_score(raw, FUMAN_COLS_2), fuman_score(raw, FUMAN_COLS_4)
    out["不満2"], out["不満4"] = f2.values, f4.values
    for name, v in _ratios(raw).items():
        out[name] = np.asarray(v, dtype=float)
        out[f"不満4x{name}"] = np.asarray(f4.values * np.asarray(v, dtype=float),
                                         dtype=float)
    return out


def cross_feature_report(raw, y):
    """事前診断用: 各列の単体AUC/AP/AP倍と、E1・E2の生列との最大|Spearman|。

    足切りには使わない（exp025 でスクリーニングとして機能しなかった）。
    移植元の数値が再現しているかの確認と、記録のために出す。
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = add_cross_features(raw, pd.DataFrame(index=raw.index))
    # E1(財務・規模) と E2(アンケート) が既に持っている生列
    base_cols = [
        "無形固定資産変動(ソフトウェア関連)", "売上", "総資産", "営業CF", "自己資本",
        "短期借入金", "長期借入金", "営業利益", "経常利益", "当期純利益", "従業員数",
    ] + [f"アンケート{k}" for k in
         ["１", "２", "３", "４", "５", "６", "７", "８", "９", "１０", "１１"]]
    base = raw[base_cols].astype(float)

    rows = []
    for c in ALL_CROSS_COLS:
        v = X[c].astype(float)
        m = v.notna().values
        auc = roc_auc_score(y[m], v[m])
        ap = average_precision_score(y[m], v[m])
        flip = auc < 0.5
        if flip:
            auc = roc_auc_score(y[m], -v[m])
            ap = average_precision_score(y[m], -v[m])
        r = max(abs(spearmanr(v, base[b], nan_policy="omit").statistic)
                for b in base_cols)
        rows.append(dict(列=c, AUC=auc, AP=ap, AP倍=ap / y.mean(),
                         最大r=r, 符号反転=flip))
    return pd.DataFrame(rows).sort_values("AP", ascending=False)


def quadrant_table(raw, y, ratio="ソフト投資_売上比"):
    """田中さん nb006/nb008 の4象限表（中央値分割）を再現する。

    加法予測との差＝交互作用の大きさ(pt)も返す。移植元との照合用。
    """
    X = add_cross_features(raw, pd.DataFrame(index=raw.index))
    f, r = X["不満4"], X[ratio]
    hi_f, hi_r = f > f.median(), r > r.median()
    cell = {(a, b): float(y[(hi_f == a) & (hi_r == b)].mean())
            for a in (False, True) for b in (False, True)}
    additive = (cell[(False, False)] + (cell[(False, True)] - cell[(False, False)])
                + (cell[(True, False)] - cell[(False, False)]))
    return cell, (cell[(True, True)] - additive) * 100
