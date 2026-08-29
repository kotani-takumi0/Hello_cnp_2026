# 評価指標・損失関数の監査とH64校正実験 — 2026-08-27

## 0. 結論

評価をF1/AUC/APだけで終えず、順位・確率校正・閾値後の混同行列へ分解した。その結果、
改善余地はLightGBMのclass weightやpairwise lossではなく、**APで正則化強度を選んだmetaの
確率温度と`half_f1`則の不整合**にあった。

- LightGBMのAP early stopping: E1だけ改善し、E0/E6と最終8本合成は悪化。STOP。
- E0だけ正例重み1.5: fresh 5repで最終ΔAP `-0.00229`（0/5）。REJECT。
- pairwise logistic meta: Platt済みpointwise meta比で15rep ΔAP `+0.00050`、
  ΔF1 `-0.00087`。REJECT。
- **pointwise logloss meta + 非正則化Platt MLE**: fresh 15repでF1 `+0.01580`、
  logloss `-0.03232`、Brier `-0.00850`。校正傾きは`1.85 -> 1.03`。ACCEPT。

既存本命・既存提出は上書きしていない。校正版は独立したchallengerとして生成した。

```text
submission/submission_ensemble_experts_seed42_aauto_concatemb_crosslr_
lin003_h2h7both4v2eq_plattmle_halff1.csv
```

## 1. 監査した指標

`exp/metric_suite.py`を追加し、strict holdout Bで次を同時に記録した。

- 順位: ROC-AUC、Average Precision
- 確率: logloss、正例/負例別logloss、Brier score
- 校正: calibration intercept/slope、10分位ECE
- 二値判定: F1、precision、recall、specificity、MCC、TP/FP/FN/TN、予測正例率
- 運用パラメータ: Aだけで決めた`half_f1`閾値

F1だけでは、順位改善・温度補正・単なる閾値移動を区別できない。APだけでは最終閾値の
転送性能を見落とす。この分解を今回の採否に使った。

seed 42の現行metaは、全742行OOFでは平均予測確率24.41%に対して実正例率24.12%、
校正傾き1.06であり、一見よく校正されていた。一方、A=482から再学習してB=260へ転送する
strict holdoutでは、repによってmetaのスコア温度が大きく縮んだ。これが`half_f1`の
閾値転送を不安定にしていた。

## 2. LightGBM側のloss/metric

### 2.1 AP early stopping

現行LightGBMはbinary loglossを目的関数・early stopping指標に使う。最終選択はAP/F1なので、
seed 200でearly stoppingだけをAverage Precisionへ変更した。

最終8本合成は、現行比でAP `-0.0092`、`half_f1` F1 `-0.0047`。E1は改善したが、
E0とE6が悪化した。strict学習ではA内部の約77行だけで木数を決めるため、階段状のAPは
停止指標としてloglossより不安定だった。単一repのスモーク時点でSTOPした。

### 2.2 正例重み付きlogloss

全LightGBMへ`scale_pos_weight=1.5`を付けるスモークでは、E0だけに改善の兆候があった。
そこでE0だけを変更し、未使用seed 201〜205で現行と完全にペア比較した。

| 最終meta_8本 | 平均差 | 改善rep |
|---|---:|---:|
| AUC | -0.00035 | 1/5 |
| AP | **-0.00229** | **0/5** |
| F1 (`half_f1`) | +0.00010 | 1/5（3同値） |

閾値側ですでに不均衡へ対応しているため、学習損失側の追加補正は順位を改善しなかった。
REJECTとした。

## 3. meta lossの比較

### 3.1 設計

`exp/meta_pairwise.py`と`exp/compare_meta_losses.py`を追加した。8 Expertの予測をrep内で共有し、
次の3方式だけを差し替えた。

1. `pointwise_logloss`: 現行の非負logit meta。
2. `pointwise_logloss_platt`: 1へ非正則化2パラメータPlatt MLEを追加。
3. `pairwise_logistic_platt`: 全正例–負例ペアのlogistic ranking lossで非負重みを学習し、
   同じPlatt MLEで確率へ戻す。

alpha選択、重み学習、校正、閾値決定はすべてAの中だけで行った。Bラベルは最後の指標計算に
しか使っていない。seed 206〜210をscreen、未使用だったseed 211〜225を15rep confirmationに
使った。

### 3.2 fresh 15repの絶対値

| 指標 | 現行pointwise | pointwise + Platt | pairwise + Platt |
|---|---:|---:|---:|
| AUC | 0.95571 | 0.95571 | 0.95622 |
| AP | 0.88689 | 0.88689 | 0.88739 |
| logloss | 0.25899 | **0.22668** | 0.22562 |
| Brier | 0.07860 | **0.07010** | 0.07001 |
| F1 | 0.77968 ± 0.04977 | **0.79548 ± 0.02685** | 0.79461 ± 0.02818 |
| precision | 0.82163 | 0.79715 | 0.79134 |
| recall | 0.74921 | **0.79471** | 0.79894 |
| MCC | 0.71885 | **0.73059** | 0.72899 |
| 予測正例率 | 0.22179 | 0.24179 | 0.24487 |
| calibration slope | 1.85277 | **1.03081** | 1.03329 |
| 10分位ECE | 0.06952 | **0.03255** | 0.03191 |

### 3.3 Platt MLEの純増

`pointwise_logloss_platt - pointwise_logloss`:

| 指標 | 平均差 | rep内訳 |
|---|---:|---:|
| F1 | **+0.01580** | 7勝 / 5同値 / 3敗 |
| recall | **+0.04550** | 8勝 / 7同値 / 0敗 |
| precision | -0.02448 | 1勝 / 5同値 / 9敗 |
| MCC | +0.01175 | 7勝 / 5同値 / 3敗 |
| logloss | **-0.03232** | 12改善 / 3悪化 |
| Brier | **-0.00850** | 12改善 / 3悪化 |

paired bootstrapによる平均ΔF1の95%区間は`[+0.00233, +0.03211]`、符号反転による
片側randomization p値は`0.0303`。平均だけでなく、F1のrep間標準偏差をほぼ半減させた。

Platt変換は単調なのでAUC/APを一切変えない。改善は順位探索ではなく、縮んだ確率温度を戻して
precision寄りだった判定をrecall側へ適正化した結果である。

過去H63の`LogisticRegression(C=1)`版とは異なり、今回は校正器にL2を掛けないMLEである。
alphaをAPで選ぶと重みの全体スケールは採否に寄与せず、zero-target L2によって温度だけが
縮み得る。校正器まで1へ縮めるとその補正を妨げるため、2パラメータMLEを事前固定した。

### 3.4 pairwise lossの純増

`pairwise_logistic_platt - pointwise_logloss_platt`:

| 指標 | 平均差 | 改善rep |
|---|---:|---:|
| AUC | +0.00052 | 10/15 |
| AP | +0.00050 | 12/15 |
| F1 | **-0.00087** | 6/15（4同値） |
| logloss | -0.00106 | 10/15改善 |

APの正率だけは高いが、平均差は既存の明確改善基準`+0.005`の10分の1で、AUC正率も
70%基準（11/15）に1つ届かず、F1純増もない。pairwise lossはREJECTした。

## 4. seed 42提出候補への反映

現行seed 42の全OOFへMLE Plattをfitした。

```text
scale  = 1.0595987167
offset = -0.0133480132
```

| 項目 | 現行 | MLE Platt |
|---|---:|---:|
| OOF AUC | 0.9645 | 0.9645 |
| OOF AP | 0.9012 | 0.9012 |
| OOF F1 | 0.8305 | 0.8305 |
| OOF logloss | 0.206392 | 0.206045 |
| OOF Brier | 0.063148 | 0.063077 |
| calibrated `half_f1` | — | 0.415254 |
| raw score等価閾値 | 0.415254 | 0.423008 |
| test正例数 | 207 | 204 |

提出ラベル差は3行だけで、`843`、`851`、`1377`が`1 -> 0`。`0 -> 1`はない。
CSVは800行・ID順一致・0/1値を検証済み。外部提出は行っていない。

## 5. 最終判断

### ACCEPT

- 評価時に`metric_suite.py`の順位・確率・閾値指標を併記する。
- 現行pointwise metaの後段に、非正則化Platt MLEを置くchallenger。
- 上記校正版CSVを第2候補として保持する。

### REJECT / STOP

- LightGBMのAP early stoppingへの一律変更。
- 今回検証したclass weight / `scale_pos_weight=1.5`。
- pairwise logistic meta loss。
- cached OOFで採用基準未達だったpairwise混合lossの追加探索。
- soft-F1 25%混合。後続の拡張監査5repでΔF1 `-0.00417`となりREJECT。

本命CSVは上書きしない。校正版は順位を変えず3行だけを動かすため、モデル多様性枠ではなく
**同一モデルのdecision/calibration hedge**として扱う。

## 6. 後続の損失family監査

Focal、Brier、weighted LogLoss、soft-F1混合、pairwise混合、uniform縮小を追加した
screen 5rep + fresh confirmation 15repの結果は
`documents/meta_loss_family_audit_20260827.md`を参照。通常pointwise LogLossを置き換える
候補はなく、本資料の本命判断は変更しない。
