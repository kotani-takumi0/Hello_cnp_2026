# H66 反実仮想ツイン差分前処理 — 2026-08-29

## 結論

現本命8 Expertに、同業・同規模の購入企業/非購入企業との差を5列へ圧縮した
Counterfactual Twin Expertを9本目として追加した。

fresh seed 500〜504のstrict完全ホールドアウト5repでは、8本基準に対して

- ΔAUC `+0.00068`（3/5勝）
- ΔAP `-0.00048`（2/5勝）
- ΔF1 `+0.00371`（2/5勝）
- 曖昧帯ΔAUC `+0.00431`（3/5勝）
- 曖昧帯ΔAP `-0.00171`（2/5勝）

となった。既存Expertとの最大Spearmanはrep平均で約0.59、meta重みも平均11.95%あり、
**異なる順位を作ること自体には成功**した。しかし主判定のAPが負で、勝率も再現条件を
満たさない。判定は **REJECT（ノイズと区別できず）**。15rep confirmationおよび
LLM比較版へは進めない。

## 1. 事前仮説

現本命の誤り61件中53件が`0.1 < p < 0.9`の228社に集中する。これらは中規模・BtoB・
DX関連業界という「あらゆる軸の中間」に位置し、企業の絶対値を再加工するだけでは分離しにくい。

そこで各社を次の問いに変換する。

> 同じ業界・上場区分・商流・規模の企業と比べたとき、この会社は購入企業と非購入企業の
> どちらに近いか。

単なるkNN多数決ではなく、交絡条件を合わせた正例ツイン/負例ツインへの距離差を、情報源別の
5列へする。LLMを使わないMVPがstrict 5repで平均ΔAP `>= +0.005`かつ4/5勝した場合だけ、
全文比較LLM版へ進む計画とした。

## 2. 前処理

### 2.1 ツインの選択

参照学習集合内で、購入/非購入クラスごとに次のcontext距離が小さい5社を選ぶ。

- カテゴリ: `業界`（重み2）、`上場種別`、`特徴`
- 規模: `従業員数`、`売上`、`総資産`の学習fold内percentile

規模は符号付き`log1p`後にfold内経験CDFへ写す。contextは比較相手の選択だけに使い、
予測列にはしない。購入ラベルは正例/負例poolを分けるためだけに使う。

### 2.2 5つの距離margin

選ばれた正例ツインへの平均距離を`d_pos`、負例ツインへの平均距離を`d_neg`として、
各viewで次の有界marginを作る。

```text
margin = (d_neg - d_pos) / (d_neg + d_pos)
```

正なら購入ツインに近い。

| 列 | 入力 | 距離 |
|---|---|---|
| `twin_margin_finance` | 安定財務比・ソフト投資・CFの11列 | median/IQR + asinh後のEuclidean |
| `twin_margin_survey` | アンケート11問 | 既知尺度へ0〜1化後のEuclidean |
| `twin_margin_dx` | DX展望embedding 1024次元 | cosine |
| `twin_margin_org` | 組織図embedding 1024次元 | cosine |
| `twin_margin_overview` | 企業概要embedding 1024次元 | cosine |

Q7の構造欠損は既存E7と同じ中立値3.3へ置く。財務比は不安定な`負債比率`と`ROE`を使わず、
`負債資産比率`、`FCF比率`、`現金ROA`等を使う。

### 2.3 リーク防止

1. outer A/B分割後、Bラベルは最終採点まで一切参照しない。
2. A内のExpert OOFを5foldで作る。
3. Twin Expertの5列headを学習する際は、さらにtrain slice内で5fold cross-fitする。
4. したがって各行の5列は、その行自身もその行のラベルもツイン参照集合に含まない。
5. 単体テストとして、予測対象20社のラベルをすべて反転しても予測最大差`0.0`を確認した。

## 3. feature単体診断

seed 500のclean 5fold OOFで、各marginを単独スコアとして測った。

| margin | AUC | AP |
|---|---:|---:|
| finance | 0.7116 | 0.4285 |
| survey | 0.5709 | 0.2806 |
| DX | **0.7538** | **0.4863** |
| organization | 0.6049 | 0.3252 |
| overview | 0.6284 | 0.3226 |

DXとfinanceには明確な単体信号がある。5列間相関はorg-overviewの0.539を除き最大0.172で、
単純な重複圧縮にはなっていない。5列LRのstrict B単体性能は5rep平均AUC 0.8142 / AP 0.6225。

## 4. strict holdout 5rep

実行:

```bash
.venv/bin/python exp/compare_counterfactual_twin.py \
  --n-reps 5 --seed-offset 500 \
  --output exp/holdout_counterfactual_twin_seed500_5rep.csv
```

| 指標 | 8 Expert | 9 Expert + Twin | ペア差 | 勝/敗 |
|---|---:|---:|---:|---:|
| AUC | 0.95559 | 0.95626 | +0.00068 | 3/2 |
| AP | 0.88269 | 0.88222 | **-0.00048** | 2/3 |
| F1 (`half_f1`) | 0.78384 | 0.78755 | +0.00371 | 2/3 |
| 曖昧帯AUC | 0.84770 | 0.85201 | +0.00431 | 3/2 |
| 曖昧帯AP | 0.83367 | 0.83196 | **-0.00171** | 2/3 |

rep別ΔAP:

```text
seed 500  -0.00540
seed 501  +0.00178
seed 502  -0.00624
seed 503  +0.00912
seed 504  -0.00164
```

Twinのmeta重みは全repで正（0.064〜0.174、平均0.1195）。最大|Spearman|も
0.530〜0.634なので、metaは独立信号として利用した。しかし正しい残差になるrepと、既存順位を
壊すrepが分かれた。

## 5. 解釈と採否

仮説の前半は成立した。

- matched positive/negative comparisonは、既存Expertと異なる順位を作れる。
- DX margin単体AUC 0.754で、企業属性に条件付けたテキスト距離にも信号はある。
- metaはTwinをゼロ重みにせず、平均12%利用した。

しかし仮説の後半、すなわち「その独立順位が既存合成の誤りを安定して直す」は成立しなかった。
全体APも曖昧帯APも負であり、成功したseed 503だけを根拠に近傍数・距離重み・列選択を調整すると
fresh holdoutへの適応になる。

したがって次を固定する。

- **REJECT**: 現行5列Counterfactual Twin Expertの本命追加。
- **STOP**: k、head C、context weight、view列の事後探索。
- **STOP**: 事前ゲート未達のため、同じツイン設計へ高コストLLM比較を追加しない。
- **KEEP**: 実装と結果CSVは、将来新しい情報源を得た場合の比較基盤として残す。

## 6. 成果物

- `exp/counterfactual_twin_features.py`
- `exp/compare_counterfactual_twin.py`
- `exp/holdout_counterfactual_twin_seed500_5rep.csv`

現本命・既定オプション・提出CSVは変更していない。
