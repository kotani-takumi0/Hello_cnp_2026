# H69: contrastive residual expert

## 結論

**REJECT。直交化はH67生追加の害を消したが、baselineを改善する独立信号はなかった。**

- テキスト単体残差: `AUC 0.45976 / AP 0.23493`。
- strict H69追加: `ΔAUC +0.00005 / ΔAP +0.00009 / ΔF1 ±0.00000`。
  4/5 seedで重み0の完全tie、1/5だけ微改善。
- 同じfresh splitでのH67生追加: `ΔAUC -0.00122 / ΔAP -0.00195 /
  ΔF1 -0.00331`。
- H69はH67追加より全5 seedでAUC/APが上だったが、これは悪い候補をゼロ棄却した効果。
  現行8本に対する改善の証拠ではない。

H67差し替えの過去の微小な弱いACCEPTを本番昇格させない判断は維持する。H67/H68/H69で
「定型文を除く・弱める・残差化する」の三方向を検証できたため、このテキスト前処理系列は
ここで閉じる。

## 仮説

既存metaは各expertのlogitへ非負重みを掛ける。したがってH67の固有成分が

`logit(H67) - beta × logit(E3)`

にある場合、meta自身はE3へ負係数を付けられず、その残差を利用できない。H69ではこの差を
fold内で明示的に作り、符号付きの1変数LRで通常の確率expertへ変換する。

## 二重cross-fit

評価target foldごとに次を行う。

1. reference行をさらに内側5-foldへ分割する。
2. 各内側foldで原文E3とH67 hard anti-templateを学習し、reference OOF確率を作る。
3. ラベルを使わず、次のOLSをreference OOF上でfitする。

   `logit(H67) = intercept + slope × logit(E3) + residual`

4. residualを標準化し、referenceラベルだけで1変数LR (`C=1.0`) をfitする。
   係数は正負どちらも許す。
5. reference全体で学習したE3/H67のtarget予測へ同じ投影・較正を適用する。
6. 得られたH69確率を9本目として既存の非負metaへ渡す。

outer B予測では、A内のE3/H67 OOFを投影・較正の学習に使い、A全体で学習したE3/H67の
B予測へ適用した。Bラベルは最終採点以外で不使用。

## 安全性・実装検査

- target側ラベル全反転による最大確率差: `0.0`
- 同一入力の再実行差: `0.0`
- 予測: 全件finite
- OLS residualとreference E3 logitのPearson相関: `6.4e-16`（スモーク）
- strict 5 seedの最終fitでも絶対相関はすべて `1e-15` 未満
- 既存8 expert予測をbaseline/H67/H69で共有

直交化とリーク防止は意図どおり動作した。

## 1. テキスト単体OOF

fresh seed 690--694。各outer foldのH69は、そのreference内でさらに5-foldを回した。

| model | AUC mean±std | AP mean±std |
|---|---:|---:|
| 原文E3 | 0.78674 ± 0.00130 | 0.51200 ± 0.00530 |
| H67 hard | 0.75138 ± 0.00710 | 0.46770 ± 0.01105 |
| H69 residual | **0.45976 ± 0.03974** | **0.23493 ± 0.02141** |

H69と原文E3のSpearmanは `-0.02826 ± 0.04797` で、予測空間でもほぼ独立になった。
しかし較正係数は `-0.02395 ± 0.04089`、絶対値平均も0.07203に過ぎず、符号が安定しない。
独立性は作れたが、購入ラベルとの関係がなかった。

## 2. strict outer holdout

- outer B=35%、fresh seed 620--624
- A内5-fold expert OOF、H69内でさらに5-fold
- Aだけでmeta alpha、重み、`half_f1` 閾値を決定

| config | AUC | AP | F1 | ΔAUC | ΔAP | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| 現行8本 | 0.95061 | 0.87281 | 0.76652 | - | - | - |
| H67生追加 | 0.94938 | 0.87087 | 0.76321 | **-0.00122** | **-0.00195** | **-0.00331** |
| H69残差追加 | 0.95066 | 0.87290 | 0.76652 | +0.00005 | +0.00009 | ±0.00000 |
| H69単体 | 0.45182 | 0.22570 | 0.39022 | - | - | - |

H69のseed別meta重みは `[0, 0, 0, 0.06465, 0]`。唯一使われたseed 623だけ
`ΔAUC +0.00024 / ΔAP +0.00043`、残りはbaselineと完全一致した。形式判定は
`REJECT (ノイズと区別できず)`。

H67生追加のmeta重みは `[0.01105, 0.02303, 0.50498, 0, 0.04281]`。特にseed 622で
誤って大きな重みを与え、F1を0.01653落とした。H69の直交化＋再較正はこの不安定な重みを
防いだが、候補を追加しないbaselineと同じ場所へ戻っただけである。

## 解釈と次の方向

H67の予測には原文E3と異なる変動がある。しかしその変動は、原文E3を線形に除いた後では
購入ラベルと安定して結び付かない。「テキストの定型部分がノイズ」という中心仮説は棄却する。

次は文章の削減・重み付けを続けず、表形式へ移る。候補は **業界条件付きpeer-relative
copula前処理**。財務値をraw比率だけでなく「同業・同上場区分の何percentileか」へfold内で
変換し、小標本groupはglobal CDFへ縮小する。絶対規模と同業内の異常度を分離でき、今回の
テキスト系列とは独立な仮説になる。

## 成果物

- `exp/contrastive_residual_text_features.py`
- `exp/compare_contrastive_residual_text.py`
- `exp/oof_contrastive_residual_text_seed690_5rep.csv`
- `exp/holdout_contrastive_residual_text_seed620_5rep.csv`
