# H56: E1財務ブロックPCA圧縮の監査 — 2026-08-27

## 結論

`documents/signate_information_quality_research.md`（section 24）で優先度4位に
位置づけられながら未実施だったH56（E1の相関ブロックをPCAで圧縮）を実装・screeningした。
5repでΔAP `+0.00151`（3勝/0同値/2敗）、ΔF1 `-0.00111`（2勝/0同値/3敗）と、指標間で
符号がばらつく小さな差にとどまり、確立済みの「明確改善」基準（ΔAP `>= +0.005`）に
遠く届かなかった。H52/H54と同じ**中立**判定でSTOPし、15rep confirmationへは進めない。

## 背景

`documents/h52_h60_and_h2_leverage_20260826.md`のH54で、E1の生財務19列
(`FINANCE_COLS`)にtarget-freeなSpearman相関（signed-log変換後）で2つのほぼ重複
ブロックが見つかっていた。

```text
size block:   総資産 ≒ 流動資産 ≒ 固定資産 ≒ 負債 ≒ 純資産 ≒ 自己資本 (rho >= 0.991)
profit block: 営業利益 ≒ 経常利益 ≒ 当期純利益 ≒ 営業CF          (rho >= 0.993)
```

H54はこれを「削除」（欠損率最小・数値安定性・解釈可能性の順で各ブロック1列だけ残す）
で対処し、Development/Audit二段階ともΔAP `-0.0014`で中立REJECTだった。削除は
落とした列固有の分散を完全に捨てる非可逆操作であり、H56（PCA、Hotelling 1933）は
その分散を主成分として残したまま次元を落とせる文献標準の代替手法として計画されていたが、
実行されていなかった。

## 実装

`exp/finance_pca_features.py`に、fold-safe（学習fold内だけでfit、target未参照）な
ブロックPCAを実装した。

```text
1. block内の各列を欠損は学習foldの中央値で埋める
2. sign(x)*log1p(|x|) で符号付き対数変換（H61で確認済みの歪度是正と同じ変換）
3. 学習foldの平均・標準偏差で標準化
4. PCA(n_components=2) を学習foldだけでfit、両foldへtransform
```

sizeブロックは2主成分で分散の99.2%、profitブロックも同程度を説明しており、圧縮の
妥当性を確認した（`documents/`記載の相関の強さと整合）。

`exp/compare_e1_pca.py`で、E1のブロック外の残り列（`資本金`・`短期借入金`・
`長期借入金`・`売上`・`減価償却費`・`運転資本変動`・`投資CF`・`有形固定資産変動`・
`無形固定資産変動`）+ `SCALE_COLS` + `H1_COLS` + `FINANCE_ISNA_COLS`はそのまま残し、
size/profitブロックだけを4列のPCAスコアへ置換した。E1のLightGBM学習自体
(`_strict_lgbm_predict`)は変更しておらず、入力特徴だけを差し替えた。他7 Expert・
outer split・inner fold・メタ損失（通常logloss）・alpha grid・cross-fitted Platt・
`half_f1`則はすべて現行本番構成と共有した。

## Screening 5rep（fresh seed 281〜285）

| 指標 | baseline (raw) | PCA候補 | Δ | 勝/同/敗 |
|---|---:|---:|---:|---:|
| F1 | 0.79172 | 0.79060 | -0.00111 | 2/0/3 |
| AP | 0.89656 | 0.89807 | +0.00151 | 3/0/2 |
| AUC | 0.95796 | 0.95820 | +0.00024 | 3/0/2 |
| LogLoss | 0.22241 | 0.22102 | -0.00139（改善） | 3/0/2 |
| MCC | 0.72668 | 0.72489 | -0.00179 | 2/0/3 |
| outer oracle F1 | 0.81391 | 0.81844 | +0.00453 | 3/0/2 |

## 解釈

H52/H54と同じ「中立」パターンが再現した。E1のLightGBMは木モデルであり、相関の強い
生列をそのまま与えても分割点として使い分けられるため、削除（H54）でもPCA（H56）でも、
どちらの圧縮方向も最終合成へは伝わらなかった。今日のH61（E0bのsigned-log、線形モデル）
で見た「単体Expertの改善が最終合成のAP/AUC純増を必須ゲートとして満たさないと採用しない」
という2.3節の原則がここでも成立する——ただし今回は単体E1改善すら明確ではなく（AP/AUC/
LogLossは僅かに改善、F1/MCCは僅かに悪化）、そもそも入力として通す価値のある新情報が
あったのか自体が疑わしい。

`documents/strategy_information_quality_over_quantity.md`（メモリ）で訂正済みの
教訓——「ノイズ削減が効くかどうかは列ごとの実際の情報量次第」——が、E1に関しては
「削除」でも「圧縮」でも同じ結論に達したことで補強された。

## 採否

- **REJECT（中立、confirmationへ進めない）**: E1財務ブロックのPCA圧縮。
- **KEEP**: E1は現行の生19列（+ SCALE/H1/ISNA）構成を維持。
- **実装として保持**: `finance_pca_features.py`・`compare_e1_pca.py`は再利用可能な
  形で残す。

この監査から新しい提出CSVは作らない。研究計画のH56はこれで完了とし、次点のH59
（E6 nested stability selection）・H60（E1 DAE）は、H52/H54/H56が揃って中立だった
時点で期待値がさらに下がっているため、優先度は低い。
