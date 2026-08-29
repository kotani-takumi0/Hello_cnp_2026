# 平滑化F1閾値則の監査 — 2026-08-27

## 結論

損失関数ではなく**閾値選択**にF1近似を適用した。校正済み確率`p`に対し
`sigmoid(k*(p-t))`でF1(t)曲線を滑らかにし、その連続argmaxを閾値とする
`smooth_f1_k*`則を実装した。screening 5repでは最も強く平滑化した`k=15`が
明確な改善（ΔF1 `+0.00522`、4勝/1同値/0敗）を示したが、fresh confirmation
15repで逆転し、**通常`half_f1`則に対しΔF1 `-0.00860`（4勝/2同値/9敗）で
有意に悪化**した（paired bootstrap 95% CI `[-0.01574, -0.00174]`、
片側sign-flip p `0.019`）。REJECT。

これで本日の「F1をどこかで近似する」試みは、メタ損失層4種・Expert損失層2種・
閾値選択層1種の**計7実装すべてがSTOP/REJECT**となった。

## 実装

`exp/compare_threshold_smoothf1.py`に、現行生産構成（`compare_meta_losses.py`の
`_inputs()`、非負pointwise LogLoss + 非正則化Platt MLE）で作った校正済みOOF確率
`pA`に対し、次で連続的な閾値を選ぶ`_smooth_f1_threshold`を追加した。

```text
S(p; t, k) = sigmoid(k * (p - t))          # 1[p>=t] の緩和
soft_tp(t) = sum(y * S), soft_f1(t) = 2*soft_tp(t) / (S.sum() + sum(y))
t* = argmax_t soft_f1(t)   （scipy.optimize.minimize_scalar, bounded [0,1]）
```

`k`は`exp/lgbm_f1_objectives.py`のsigmoidF1と同じ緩和だが、ここでは学習ではなく
**すでに固定された確率**に対する事後の閾値探索にのみ使う。`k`が大きいほど離散
argmax（`oof_best`）に近づき、小さいほど強く平滑化される。`k=15/40/100`の3候補を
試した。参考として既存の`oof_best`・`plateau_mid`則も同じ枠組みで再評価した。

## Screening 5rep（fresh seed 261〜265）

| ルール | ΔF1 vs half_f1 | 勝/同/敗 | ΔMCC |
|---|---:|---:|---:|
| `smooth_f1_k15` | **+0.00522** | **4/1/0** | +0.00781 |
| `smooth_f1_k100` | -0.00304 | 2/0/3 | -0.00165 |
| `smooth_f1_k40` | -0.00463 | 2/0/3 | -0.00435 |
| `plateau_mid` | -0.00979 | 2/0/3 | -0.01027 |

`smooth_f1_k15`だけが確立済みの「明確な改善」基準（ΔF1 `+0.005`）にちょうど届き、
confirmationへ進めた。

## Confirmation 15rep（fresh seed 266〜280）

| ルール | ΔF1 vs half_f1 | 勝/同/敗 |
|---|---:|---:|
| `smooth_f1_k100` | -0.00751 | 7/1/7 |
| `smooth_f1_k15` | **-0.00860** | **4/2/9** |
| `smooth_f1_k40` | -0.01130 | 6/1/8 |
| `oof_best` | -0.01313 | 5/1/9 |
| `plateau_mid` | -0.01390 | 4/1/10 |

`smooth_f1_k15 - half_f1`のΔF1: 平均`-0.00860`、paired bootstrap 95%区間
`[-0.01574, -0.00174]`（0を含まない）、15個のpaired差に対する全`2^15`通りの
符号反転検定で片側p `0.019`。screeningの好結果は5個の偶然良かったseedによる
過学習で、fresh confirmationでは方向ごと反転した。

`half_f1`は5候補すべてに対して確認された全ての閾値則を上回った。既存の
「自由度が最小＝最も過学習しにくい」（`holdout_check.py`のdocstring）という説明が
今回も支持された。

## 解釈

`smooth_f1`は閾値の自由度を1点(t)に絞る点では`half_f1`と同じだが、
`half_f1`は「校正済み確率のF1最適閾値はF1*/2」という理論的に導出された関係
（proper scoring ruleの一致性）を使い、A側のOOF確率分布そのものには依存しない
形で閾値を決める。一方`smooth_f1`はA側の確率分布の形（平滑化されたF1曲線の
ピーク位置）に直接依存するため、Aの標本ノイズを閾値へそのまま持ち込みやすい。
`k`を下げるほど分散は減るはずだが、confirmationでは`k=15`でも`half_f1`の
分散（threshold std `0.01249`）よりまだ大きい`threshold std 0.02866`が残り、
バイアス（平滑化による系統的な閾値シフト）とのトレードオフで結局`half_f1`に
勝てなかった。

## 採否

- **REJECT**: `smooth_f1_k15`/`k40`/`k100`、`oof_best`、`plateau_mid`。
- **KEEP**: 現行`half_f1`則（校正済みF1*/2）。
- **実装として保持**: `compare_threshold_smoothf1.py`は再利用可能な形で残す。

この監査から新しい提出CSVは作らない。
