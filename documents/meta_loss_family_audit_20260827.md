# メタ損失関数の拡張監査 — 2026-08-27

## 結論

現行のメタ学習器は **通常のpointwise LogLossを維持**する。Focal、Brier、正例重み、
soft-F1混合、pairwise混合、uniform縮小を同一条件で比較したが、fresh 15反復でLogLossを
置き換えられる候補はなかった。

正例を1.5倍したweighted LogLossだけは確認試験で平均F1が`+0.00279`だったが、
95%区間は`[-0.00115, +0.00727]`、片側符号反転検定は`p=0.1328`。15反復中の内訳も
4勝/8同値/3敗であり、提出候補へ昇格させない。

したがって現時点の構成は次を維持する。

1. Expert学習: binary LogLoss。
2. Meta重み学習: 非負pointwise LogLoss。
3. 確率温度: 非正則化Platt MLE。
4. 二値化: calibrated `half_f1`則。

直接F1へ近づけた損失を使うより、proper lossで順位を学び、校正と閾値を別工程で扱う方が
この742行データでは安定した。

## 比較設計

変更したのは8 Expertをまとめるメタ目的関数だけである。特徴量、Expert予測、outer split、
inner fold、正則化グリッド、校正、閾値則はrep内で共有した。

- データ: train 742行、正例179行。
- outer holdout: 層化35%（260行）。Bのラベルは最終評価まで未使用。
- A内Expert OOF: 5-fold。
- メタ正則化: `alpha=1e-5 ... 1.0`をA内APで選択。
- メタ校正: cross-fitted marginをfold学習スコアの平均・標準偏差で揃え、Aだけで
  非正則化Plattをfit。
- 閾値: Aのcross-fitted calibrated予測から`F1*/2`を計算。
- 指標: AUC/AP、LogLoss/Brier、校正傾き/ECE、F1/precision/recall/MCC、混同行列、
  outer oracle F1。
- screening: seed 226〜230、9候補。
- confirmation: screeningを見た後に候補を固定し、未使用seed 231〜245、5候補。

Focal・weighted lossの出力は校正確率ではないため、全方式へ同じcross-fitted Plattを適用した。
これにより「温度が違うだけ」を損失関数の勝ちとして数えず、順位と最終decisionの差を比較した。

## 比較した目的関数

| 名前 | 内容 | 狙い |
|---|---|---|
| `logloss` | 通常BCE + zero-target L2 | 現行対照 |
| `logloss_uniform` | 通常BCE + uniform-target L2 | Expert配分の縮小先を修正 |
| `brier` | 二乗確率誤差 | 確率校正を重視 |
| `weighted_logloss_pos15` | 正例BCEを1.5倍 | recall不足を補正 |
| `focal_gamma1` | 対称Focal, gamma=1 | 難例を重視 |
| `focal_gamma2` | 対称Focal, gamma=2 | 難例をさらに重視 |
| `asymmetric_focal_neg2` | 正例gamma=0、負例gamma=2 | easy negativeを抑制 |
| `logloss_softf1_025` | 75% BCE + 25% soft-F1 | decision surrogateを混合 |
| `logloss_pairwise_025` | 75% BCE + 25% pairwise logistic | 順位surrogateを混合 |

全損失の解析勾配は数値微分と最大`3e-8`以内で一致した。`logloss`実装は既存
`meta_blend.fit_meta`と重み・切片・予測が完全一致した。

## Screening 5反復

| 損失 | F1 | ΔF1 vs LogLoss | AP | LogLoss |
|---|---:|---:|---:|---:|
| `logloss_uniform` | **0.80819** | +0.00486 | 0.89318 | 0.22485 |
| `asymmetric_focal_neg2` | 0.80511 | +0.00178 | 0.89411 | 0.22413 |
| `logloss` | 0.80333 | — | 0.89507 | 0.22321 |
| `brier` | 0.80327 | -0.00006 | **0.89684** | **0.22101** |
| `logloss_pairwise_025` | 0.80268 | -0.00065 | 0.89550 | 0.22292 |
| `focal_gamma1` | 0.80149 | -0.00185 | 0.89521 | 0.22298 |
| `weighted_logloss_pos15` | 0.80091 | -0.00243 | 0.89619 | 0.22284 |
| `focal_gamma2` | 0.80080 | -0.00253 | 0.89488 | 0.22329 |
| `logloss_softf1_025` | 0.79916 | -0.00417 | 0.89428 | 0.22266 |

screeningでは効き方が異なる`uniform`、Brier、非対称Focalを選び、outer oracle F1に兆候が
あったweighted LogLossも参考候補としてconfirmationへ進めた。soft-F1、pairwise混合、
対称Focalはこの時点で停止した。

## Fresh confirmation 15反復

| 損失 | F1 | AP | LogLoss | Brier | MCC | oracle F1 |
|---|---:|---:|---:|---:|---:|---:|
| `weighted_logloss_pos15` | **0.80188** | 0.89088 | **0.22055** | **0.06859** | **0.73910** | **0.82156** |
| `brier` | 0.80015 | 0.89001 | 0.22180 | 0.06913 | 0.73688 | 0.81967 |
| `logloss` | 0.79910 | 0.89088 | 0.22097 | 0.06876 | 0.73563 | 0.81998 |
| `logloss_uniform` | 0.79894 | 0.88626 | 0.22475 | 0.07009 | 0.73555 | 0.81751 |
| `asymmetric_focal_neg2` | 0.79790 | **0.89100** | 0.22065 | 0.06869 | 0.73380 | 0.81741 |

### LogLossとの差と不確実性

| 候補 | ΔF1 | paired bootstrap 95% | F1勝/同/敗 | 片側p | ΔAP | ΔLogLoss |
|---|---:|---:|---:|---:|---:|---:|
| weighted LogLoss | +0.00279 | [-0.00115, +0.00727] | 4/8/3 | 0.1328 | -0.00000 | -0.00042 |
| Brier | +0.00106 | [-0.00383, +0.00591] | 5/6/4 | 0.3359 | -0.00087 | +0.00084 |
| uniform縮小 | -0.00016 | [-0.00889, +0.00748] | 9/1/5 | 0.5138 | **-0.00462** | **+0.00378** |
| 非対称Focal | -0.00120 | [-0.00559, +0.00279] | 4/7/4 | 0.6992 | +0.00012 | -0.00032 |

`ΔLogLoss`はcandidate - baselineなので負が改善。bootstrapはseedをpaired resamplingし、p値は
15個のpaired差に対する全`2^15`通りの符号反転から算出した。

## 解釈

### weighted LogLoss

平均値は最良だが、8/15 splitで最終予測がLogLossと完全一致した。改善した4 splitでも
動く混同行列セルは1〜3件で、効果は境界点の微調整に限られる。APは実質同一で、順位全体を
改善した証拠ではない。候補として記録するが、現時点で提出・本番化しない。

### Brier

screeningの確率指標改善はconfirmationで反転した。Brierはproper lossだが、メタ重み8個の
小標本推定とL2選択を含めると、BCEより安定するとは限らなかった。REJECT。

### Focal / soft-F1 / pairwise

難例強調、F1 surrogate、ranking surrogateのいずれも最終F1を安定して上げなかった。
このパイプラインでは最後に校正とF1閾値最適化を独立して行うため、学習損失側でも同じ補正を
重ねる利点が小さい。特にsoft-F1はscreeningで最下位だった。

### uniform縮小

screeningのF1利得は再現せず、confirmationではAP `-0.00462`、LogLoss `+0.00378`、
Brier `+0.00133`。一部splitでF1だけ上がる一方、順位と確率品質を一貫して落とした。
REJECT。

## 採否

- **KEEP**: pointwise LogLoss + 非正則化Platt + calibrated `half_f1`。
- **WATCHLIST**: weighted LogLoss positive weight 1.5。追加証拠なしでは使わない。
- **REJECT**: Brier、uniform縮小、対称/非対称Focal、soft-F1混合、pairwise混合。
- **既存判断を維持**: LightGBM E0の`scale_pos_weight=1.5`はmeta APが5/5で悪化したため不採用。

この監査から新しい提出CSVは作らない。既存Platt候補も含め、提出判断を損失探索の
screening結果で変更しない。

## 後続: 反復F1 surrogate

F1の現在値から偽陰性・偽陽性コストを作り、weighted LogLossを再学習してF1を更新する
反復法も実装・screeningした。推定F1約0.8から正例重み約1.5へ収束し、通常LogLoss比の
最終ΔF1は5repで`-0.00524`だった。詳細は
`documents/iterative_f1_surrogate_audit_20260827.md`を参照。
