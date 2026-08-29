# 反復F1 surrogate監査 — 2026-08-27

## 結論

「現在のF1をcost-sensitive lossで近似し、そのF1から次のlossを作る」という反復法を
メタ学習器へ実装した。アルゴリズムは全splitで固定点へ収束したが、strict holdout 5repで
通常LogLossにF1で0勝/2同値/3敗、平均差`-0.00524`だったため、productionへは採用しない。

反復法が推定したF1は平均`0.79982`、そこから自動導出された正例重みは平均`1.50193`だった。
したがって、先に検証した「正例1.5倍LogLoss」は、この反復F1近似のほぼ1-step解になっていた。

実装は今後の拡張用に残すが、この結果から提出CSVは生成しない。

## 理論形

F1は

```text
F1 = 2 TP / (2 TP + FP + FN)
```

という分数目的である。F1の現在推定値を`q`とすると、そのlevel setに対応する
cost-sensitive分類は次の誤分類コストを持つ。

```text
false negative cost = 1 - q / 2
false positive cost = q / 2
```

weighted LogLossで負例重みを1へ正規化すると、正例重みは

```text
w_pos(q) = (2 - q) / q
```

となる。例えば`q=0.8`なら`w_pos=1.5`である。このF-measureのpseudo-linearな
cost-sensitive reductionはParambath et al. (NeurIPS 2014)に基づく。F-measureを
一連のcost-sensitive問題として解く考え方と有限標本での保証が示されている。

- https://papers.nips.cc/paper_files/paper/2014/hash/5c0314ec1b57fcd36bbb013f3f025868-Abstract.html
- https://proceedings.mlr.press/v89/bascol19a.html

今回の実装は、このreductionをweighted logistic surrogateで解き、現在F1を更新する
Dinkelbach-likeなouter loopにした。厳密な0/1 cost問題ではなくLogLoss surrogateを
内側で使うため、理論アルゴリズムそのものではなく安定した近似実装である。

## アルゴリズム

outer train Aの中だけで次を行う。

1. `q_0=0.8`から開始。
2. `w_pos=(2-q_t)/q_t`を計算。
3. その重みを持つ非負LogLossメタモデルを5-foldで学習。
4. honest OOF marginを0で二値化し、`q_observed=F1`を計算。
5. `q_{t+1}=0.5 q_t + 0.5 q_observed`で更新。
6. `|q_observed-q_t| <= 1e-5`まで反復。最大60回。
7. 収束したalpha候補だけからA内APで正則化強度を選択。
8. 最終モデルには従来どおりcross-fitted Plattと`half_f1`を適用。
9. outer holdout Bは最後の評価まで未使用。

離散F1は狭い2周期を作り得るため、2周期を検出した場合はtrajectory内の最高F1点を保持する。
今回選択された5候補はすべて通常の固定点へ収束し、周期fallbackは使われなかった。

## 実装検証

- `q=0.8 -> w_pos=1.5`を数値確認。
- seed 42、alpha 0.003で初期値を`0.50 / 0.80 / 0.95`へ変えても、同一ラベル、
  F1 `0.83799`、正例重み約`1.3867`へ収束。
- screening 5repでは全候補が10〜13 iterationで固定点へ収束。
- 記録された`positive_weight`と`(2-F1)/F1`の最大誤差は`2.91e-5`。
- 全outer splitでExpert予測をbaselineと共有。

## Strict screening 5rep

未使用seed 246〜250を使用した。

| 方法 | F1 | AP | LogLoss | Brier | MCC | oracle F1 |
|---|---:|---:|---:|---:|---:|---:|
| 通常LogLoss | **0.80048** | 0.88774 | 0.23182 | 0.07085 | **0.73902** | 0.82347 |
| 固定正例1.5倍 | 0.79955 | 0.88679 | 0.23296 | 0.07122 | 0.73768 | 0.82060 |
| 反復F1 cost LogLoss | 0.79524 | **0.88812** | **0.23058** | **0.07072** | 0.73253 | **0.82382** |

### 反復法 − 通常LogLoss

| 指標 | 平均差 | 勝/同/敗 |
|---|---:|---:|
| F1 | **-0.00524** | **0/2/3** |
| AP | +0.00038 | 4/0/1 |
| LogLoss | -0.00125 | 3/0/2改善 |
| Brier | -0.00012 | 2/0/3改善 |
| MCC | -0.00649 | 0/2/3 |
| oracle F1 | +0.00035 | 1/2/2 |

反復lossが直接決めたmargin 0の判定も平均F1 `0.79794`で、通常pipeline比`-0.00254`
（1勝/2同値/2敗）だった。後段校正だけが失敗原因ではない。

## 解釈

F1はTP/FP/FNの集計だけで決まるため、そのlevel-set近似から得られる重みはクラス単位の
2コストへ還元される。このデータでは推定F1が約0.8なので、反復しても正例重みは約1.5へ
戻った。すでに試した固定1.5倍から大きな新情報は生まれない。

APは4/5で改善したが平均`+0.00038`に留まり、decision境界の改善へ接続しなかった。
一方、proper logistic lossを学習して別標本でF1閾値を選ぶ二段階法には理論的なregret保証が
あり、今回もその現行構成の方が安定した。

- https://link.springer.com/article/10.1007/s10994-016-5591-7

## 採否

- **実装として採用**: 反復F1 cost-surrogateを再利用可能な実験モジュールとして保持。
- **モデルとしてSTOP**: screening gate未達のためfresh confirmationへ進めない。
- **提出しない**: 現行LogLoss + Platt + `half_f1`を維持。

