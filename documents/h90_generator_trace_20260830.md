# H90 通常の生成痕跡特徴量 — 2026-08-30

## 結論

**REJECT。現行8 Expertと提出物は変更しない。**

提供行だけから作る通常の生成痕跡292列を、強いL2をかけた独立LR Expertとして検証した。
単体では AUC `0.7550` / AP `0.4884` と予想外に強かったが、現行8本へ足した差は
5 seed平均で `ΔAUC -0.00027 / ΔAP -0.00040 / ΔF1 -0.00065`。
勝負所の曖昧帯も `ΔAP -0.00307` で、既存Expertが持たない正しい残差は無かった。

自然な特徴群へ分けた探索で文章書式だけ一時的に `ΔAP +0.00110` となったが、
探索に使っていない seed 5–9 では `ΔAP -0.00026 / ΔF1 -0.00090` に反転した。
丸め・末尾桁・文章レイアウトのどこにも再現可能な増分は無い。

PRNG状態の復元、Leaderboard probing、外部データ、手動ラベルは一切使用していない。

---

## 1. 事前固定した設計

### 特徴（すべてtarget-free・行単位）

| 群 | 列数 | 内容 |
|---|---:|---|
| 数値末尾 | 170 | 23数値列の末尾10進周期、偶奇、trailing zero、ゼロ・欠損、行内末尾数字分布 |
| 会計残差 | 26 | 2本の会計恒等式の丸め差、営業CF・投資CF関連列のmod 10/100/1000 |
| 文章書式 | 96 | 4文書の文字数、bytes、改行、文数、句読点、文字種比率、行長、企業名形式 |
| **全部** | **292** | 上記3群の結合 |

`企業ID` は train `0–741` / test `742–1541` で支持域が完全分離し、過去の単純周期診断も
無効だったため入力していない。生の金額や文章の単語は入れず、値の描画方法だけを見る。

### モデルと配置

```text
VarianceThreshold(1e-12)
→ StandardScaler
→ LogisticRegression(C=0.01, L2, liblinear)
```

現行8本は一切変更せず、`E13_generator_trace` として9本目へ追加する。
メタは既存と同じ非負logit合成。効かなければ重み0へ落ちる設計である。

---

## 2. 全292列のfull-data OOFスクリーニング（seed 0–4）

既存8本は凍結済みキャッシュを利用し、H90だけ同じfoldでcross-fitした。
これは高速スクリーニングであり、正式採否にはstrict外側holdoutが必要になる。

| 指標 | 現行8本 | + H90 | 差 | 改善seed |
|---|---:|---:|---:|---:|
| AUC | 0.96025 | 0.95999 | **-0.00027** | 1/5（同値1） |
| AP | 0.88690 | 0.88651 | **-0.00040** | 1/5（同値1） |
| F1 (`half_f1`) | 0.80844 | 0.80779 | **-0.00065** | 0/5（同値4） |
| 曖昧帯AUC | 0.80037 | 0.79911 | **-0.00125** | 1/5（同値1） |
| 曖昧帯AP | 0.76229 | 0.75922 | **-0.00307** | 2/5（同値1） |

H90のメタ重みは seed別に `0 / 0.0055 / 0.0061 / 0.0144 / 0`。
最大でも1.44%で、2 seedでは完全に0へ落ちた。既存Expertとの最大Spearmanは
`0.555–0.604`なので、同一予測だから消えたのではなく、**独立だが正しくない信号**だった。

### 単体性能とIDブロック転送

| 検証 | AUC | AP | F1 (`half_f1`) |
|---|---:|---:|---:|
| shuffled OOF 5 seed平均 | 0.75499 | 0.48837 | 0.48617 |
| ID連続5 block OOF | 0.75408 | 0.48677 | 0.48200 |

単体信号はIDブロックを跨いでも維持された。したがって単なる行近接リークではない。
ただし曖昧帯での単体AUCは5 seed平均 `0.5127`。全体で強く見えるのは、既存モデルも取れる
企業規模・DX成熟度・文書テンプレートの代理であり、必要な境界残差ではなかった。

---

## 3. 自然な特徴群への分解（探索seed 0–4）

全部入りで弱い痕跡が埋もれた可能性を、結果の意味に沿った3群だけで診断した。

| 構成 | 列数 | 単体AUC | 単体AP | 合成ΔAUC | 合成ΔAP | AP改善 |
|---|---:|---:|---:|---:|---:|---:|
| 数値末尾 | 170 | 0.6247 | 0.3424 | -0.00001 | **-0.00002** | 0/5 |
| 会計残差 | 26 | 0.5451 | 0.2625 | -0.00009 | **-0.00003** | 1/5 |
| 数値全部 | 196 | 0.6316 | 0.3601 | -0.00027 | **-0.00030** | 2/5 |
| 文章書式 | 96 | 0.7378 | 0.4375 | -0.00017 | **+0.00110** | 3/5 |

数値生成痕跡はこの時点で終了。文章書式だけ符号が正だったため、これだけを
未使用seedで一度確認した。

### 文章書式の未使用seed 5–9確認

| 指標 | 平均差 | 改善seed |
|---|---:|---:|
| AUC | **-0.00019** | 1/5 |
| AP | **-0.00026** | 2/5 |
| F1 (`half_f1`) | **-0.00090** | 0/5（同値2） |
| 曖昧帯AUC | **-0.00240** | 1/5 |
| 曖昧帯AP | **-0.00224** | 2/5 |

探索側の `+0.00110` は再現せず、AUC/AP/F1がすべて負へ反転した。
文章長や改行形式が持つ信号も、既存のテキスト・構造Expertへの増分ではない。

---

## 4. strict外側holdoutスモークと停止

full-dataスクリーニングが事前停止条件（平均ΔAP負）に達していたため、本来ならここで終了する。
実装配管の確認に限り、未使用seed 700でstrict A/B holdoutを1回だけ実行した。

| 指標 | 現行8本 | + H90 | 差 |
|---|---:|---:|---:|
| AUC | 0.96100 | 0.96036 | **-0.00064** |
| AP | 0.90580 | 0.90375 | **-0.00205** |
| F1 (`half_f1`) | 0.77778 | 0.77778 | 0.00000 |
| 曖昧帯AP | 0.90580 | 0.90375 | **-0.00205** |

H90重みは2.11%。方向はfull-data screenと一致して負だった。

これは1回なので単独では正式判定にならない。しかし、事前screen 5 seed、自然群分解、
文章書式のfresh 5 seed確認がすべて不合格である。既定の早期停止規則に従い、
strict 5–15 repと高コストなfull-pipeline forward検証は実行しない。

---

## 5. 解釈

生成痕跡は**存在する**。書式だけで単体AUC 0.74、全痕跡で0.75になるため、生成テンプレートは
企業のDX成熟度や業界・規模と結びついている。しかし、その情報はE0/E0b/E3/E4が既に回収している。

一方、本当に新情報になり得た末尾数字・会計丸めは、IDブロックを跨いだ単体性能こそ
偶然以上でも、現行合成への寄与が一貫して0以下だった。**通常の特徴量として利用できる
生成痕跡から、ラベル抽選残差は読めない。**

### 判定

- **REJECT**: 全292列の9本目Expert
- **REJECT**: 数値末尾・会計残差・文章書式の各独立Expert
- 提出ファイルは作らない
- `企業ID`、modulus、C、書式統計の追加探索は行わない
- PRNG復元など規約上グレーな方向へは進まない

---

## 6. 再現

```bash
.venv/bin/python -m unittest exp/test_generator_trace_features.py -v
.venv/bin/python exp/compare_generator_trace.py --mode screen
.venv/bin/python exp/compare_generator_trace.py --mode screen \
  --feature-group numeric_digits
.venv/bin/python exp/compare_generator_trace.py --mode screen \
  --feature-group accounting_residuals
.venv/bin/python exp/compare_generator_trace.py --mode screen \
  --feature-group text_layout
.venv/bin/python exp/compare_generator_trace.py --mode screen \
  --feature-group text_layout --seed-offset 5 \
  --output exp/generator_trace_screen_text_layout_seed5_5rep.csv
.venv/bin/python exp/compare_generator_trace.py --mode strict \
  --n-reps 1 --seed-offset 700 \
  --output exp/holdout_generator_trace_seed700_1rep.csv
```

成果物:

- `exp/generator_trace_features.py`
- `exp/compare_generator_trace.py`
- `exp/test_generator_trace_features.py`
- `exp/generator_trace_screen_seed0_5rep.csv`
- `exp/generator_trace_screen_numeric_digits_seed0_5rep.csv`
- `exp/generator_trace_screen_accounting_residuals_seed0_5rep.csv`
- `exp/generator_trace_screen_numeric_all_seed0_5rep.csv`
- `exp/generator_trace_screen_text_layout_seed0_5rep.csv`
- `exp/generator_trace_screen_text_layout_seed5_5rep.csv`
- `exp/holdout_generator_trace_seed700_1rep.csv`
