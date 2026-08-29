# sigmoidF1 (Bénédict et al. 2021) 損失関数の監査 — 2026-08-27

## 結論

「F1を微分可能な形で近似し、それ自体を学習lossにする」という発想のWebベストプラクティス
実装として、sigmoidF1 (Bénédict et al., *sigmoidF1: A Smooth F1 Score Surrogate Loss for
Multilabel Classification*, arXiv:2108.10566) をメタ学習器へ追加した。screening 5repで
3構成すべてが通常pointwise LogLossに対しF1で勝ち越せず、confirmationへ進めずSTOPした。

これで本日だけで独立した4種類のF1近似loss——①反復cost-sensitive法
（`documents/iterative_f1_surrogate_audit_20260827.md`）、②soft-F1 25%混合
（`documents/meta_loss_family_audit_20260827.md`）、③今回のnaive soft-F1単体
（sigmoid_f1_b1e0）、④sigmoidF1本来の学習可能な傾き・オフセット付き版——が同じ
結論に収束した。このデータ（742行、正例179）では学習lossをF1へ寄せるより、
proper loss (LogLoss) を学習し校正と閾値選択を別工程に分離する方が安定する。

## 実装

`exp/meta_loss_lab.py`に`_sigmoid_f1_margin(scores, y, beta, eta)`を追加した。

```text
S(u; beta, eta) = sigmoid(beta * (u + eta))          # 論文のsurrogate thresholding
tp~ = sum(y * S), fp~ = sum((1-y) * S), fn~ = sum(y) - tp~
F1~ = 2 tp~ / (tp~ + fp~ + sum(y))
loss = 1 - F1~
```

`beta=1, eta=0`は既存の`_soft_f1_margin`（何もスケールしない素朴なsoft-F1）と厳密に一致する
ことを確認した（loss差・勾配差ともに0）。解析勾配は数値微分と最大`9.23e-11`で一致した。

論文はこのsigmoidF1を**単独**の損失として使い、BCEとは混合しない。既存の
`logloss_softf1_025`（BCE 75% + soft-F1 25%混合、`weight_l1`最終値なし）とは異なり、
今回はBCEを一切混ぜない直接最適化として比較した。

3候補を`LOSS_SPECS`へ追加した。

| 名前 | beta | eta | 狙い |
|---|---:|---:|---|
| `sigmoid_f1_b1e0` | 1 | 0.0 | naive soft-F1をBCE混合なしの単独lossとして検証 |
| `sigmoid_f1_b7e1` | 7 | 1.0 | 論文のグリッド中央付近（beta∈[1,30], eta∈[0,2]） |
| `sigmoid_f1_b20e15` | 20 | 1.5 | ステップ関数に近い急峻な近似 |

## Screening 5rep

未使用seed 251〜255。他条件（Expert予測、outer split、inner fold、alpha grid、
cross-fitted Platt校正、`half_f1`閾値則）は`compare_meta_loss_families.py`の既存フレームで
LogLossと共有した。

| 損失 | F1 | ΔF1 vs LogLoss | F1勝/同/敗 | AP | ΔAP |
|---|---:|---:|---:|---:|---:|
| `logloss`（現行） | **0.82413** | — | — | 0.88904 | — |
| `sigmoid_f1_b7e1` | 0.82411 | -0.00002 | 3/0/2 | 0.89309 | +0.00404 |
| `sigmoid_f1_b1e0` | 0.82120 | -0.00293 | 2/0/3 | 0.89234 | +0.00329 |
| `sigmoid_f1_b20e15` | 0.81770 | -0.00643 | 1/1/3 | 0.88798 | -0.00107 |

`sigmoid_f1_b7e1`はAPをわずかに改善しF1もほぼ同点だが、5repのうち2repで負けており、
既存の「明確な改善傾向がなければconfirmationへ進めない」screening基準に届かない。
`weight_l1`・`alpha`のrep間分散も他方式よりかなり大きく（例: alpha std 0.53 vs logloss
0.005）、選択が不安定であることを示している。

## 解釈

sigmoidF1論文の主張（naive soft-F1はunbounded gradientやsaturationの問題で劣り、
tunable slope/offsetが必要）はこの実装でも整合していた——`b1e0`（naive相当）が
3候補中最もF1が悪く、傾きを持たせた`b7e1`がわずかに改善した。しかし、そのわずかな
改善も本パイプラインの最終F1を安定して超えるには至らなかった。8 Expertのメタブレンド
という低次元・小標本設定では、確率のランキング品質(LogLoss/AP)と最終decisionの分離が
既に`half_f1`則で担保されており、F1へ直接寄せた損失を追加で挟む余地が小さいという
既存3監査の解釈がここでも再現した。

## 採否

- **REJECT**: sigmoidF1（3構成とも）。confirmationへ進めない。
- **KEEP**: pointwise LogLoss + 非正則化Platt MLE + calibrated `half_f1`。
- **実装として保持**: `_sigmoid_f1_margin`は再利用可能な形で`meta_loss_lab.py`に残す。

この監査から新しい提出CSVは作らない。
