# 現行パイプライン総点検と反復実験 — 2026-08-27

## 0. 結論

今回の点検後も、本命は次の構成を維持する。

- モデル: **H2+H7-both4修正版の8 Expert非負ロジット合成**
- seed: 42
- 閾値則: **`half_f1`**
- 最終閾値: `0.4152542372881356`
- 提出ファイル: `submission/submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq_th04152542372881356.csv`
- 全742行OOF: AUC `0.9645` / AP `0.9012` / F1 `0.8305`
- test予測正例: 207 / 800（25.875%）
- Public F1: `0.8000`

今回新たに、前処理・モデル編成・確率校正を順番に検証した。

1. E0bの金額・規模23列を符号付き対数へ変換すると、E0b単体は明確に改善した。
2. しかし8本合成のAUC/APは改善せず、raw版とlog版の二重化もF1を悪化させた。
3. Platt校正後の`half_f1`はfresh 15分割で良好だったが、未校正版`half_f1`への純増は
   4勝4敗7同値で、2パラメータ増やす根拠にはならなかった。
4. 現行`half_f1`は、開発15分割で`oof_best`比F1 `+0.00947`、完全に未使用の
   fresh 15分割でも`+0.00826`となり、平均改善量を独立に再現した。

したがって、**新しいコードは再現可能な実験オプションとして残すが、既定値・本命モデル・
提出物は変更しない**。Public値を見た追加調整も行わない。

---

## 1. データと評価原則

### 1.1 データ

| 項目 | 値 |
|---|---:|
| train | 742行 |
| train正例 | 179行（24.124%） |
| test | 800行 |
| 目的変数 | `購入フラグ` |
| 評価指標 | F1 |

742行に対して特徴・モデル・閾値の自由度が大きいため、単一seedのOOF最大値は採用根拠にしない。
順位の主判定をAP、検出補助をAUC、最終ガードレールを転送F1とする。

### 1.2 正式比較

`exp/holdout_check.py`のstrict完全ホールドアウトを使う。

1. 層化分割でA=482行、B=260行に分ける。
2. A内5-foldだけで各ExpertのOOFを作る。
3. AのOOFだけでmetaのalpha・重み・閾値・校正器を決める。
4. ExpertをA全体で再学習し、Bを予測する。
5. Bラベルは最後のスコア計算まで一切使わない。

LightGBMの木数もA内部の固定holdoutで決め、A全体へrefitする。Bラベルをearly stoppingへ
渡した旧実装は採否に使用しない。同一rep内の差分だけを読み、絶対F1の分割間差をモデル差と
誤認しない。

判定基準は既存の`exp/decision.py`を維持する。

- 平均ΔAP `>= +0.005`を明確改善の目安とする。
- AUC改善率70%以上、AP改善率75%以上を安定性の目安とする。
- 平均ΔF1 `< -0.002`はREJECTする。
- 単体Expertの改善だけでは採用せず、最終合成への増分を見る。

---

## 2. 現行パイプライン

### 2.1 前処理

構造化データの基本処理は次のとおり。

- `企業ID`、`企業名`、3本文列を表形式モデルの入力から除外する。
- `事業所数`、`工場数`、`店舗数`、`アンケート７`は0埋めし、欠損フラグも追加する。
- `資本金`、`営業利益`、`経常利益`は値を欠損のまま保ち、欠損フラグを追加する。
- `業界`、`上場種別`、`特徴`はカテゴリとして扱う。
- 利益率、組織図/DX展望/企業概要の固定辞書特徴、LLM Make-or-Buy親4軸を生成する。
- H2+H7では`自己資本比率`、`ROA`、`FCF比率`、`現金ROA`をE0/E0bへ明示的に配線する。
- TF-IDFスコアはfold学習側だけでfitし、validationへ転送する。
- E0bのカテゴリはfold学習側の列へone-hot整列し、数値欠損はfold学習側中央値で埋める。

現在の決定的特徴行列は128列。fold内テキストスコアを加えた各入力列数は次のとおり。

| Expert | 入力列数・表現 | モデル |
|---|---|---|
| E0 anchor | 71決定列 + DX/組織スコア2列 = 73 | LightGBM |
| E1 finance | 財務・規模32列 | LightGBM |
| E2 survey | アンケート12列 | Logistic Regression |
| E3 dx_text | `今後のDX展望` TF-IDF | Logistic Regression |
| E4 org_text | 組織図1024次元 + 企業概要1024次元 | Logistic Regression |
| E6 manual | 手動辞書・構造48列 | LightGBM |
| E7 cross | 不満・財務の親/交互作用12列 | Logistic Regression |
| E0b linear | 96決定列 + DX/組織スコア2列 = 98 | one-hot + StandardScaler + LR (`C=0.03`) |

### 2.2 合成

8 Expertの確率をlogitへ変換し、非負重み付きlogistic metaで合成する。alphaは外側trainの
内側CVでAPにより選び、評価foldは重み・alpha選択に使わない。seed 42全OOFでの平均重みは、
E2、E0、E4、E0b、E7、E1、E6、E3の順で、E3は0になった。ただし重み0はfold・seed依存で、
過去の完全ホールドアウトではE3/E4削除が安定改善しなかったため、8本を維持する。

### 2.3 閾値

全OOFで得たF1 `0.830508...`の半分を閾値とする。

```text
threshold = OOF達成F1 / 2 = 0.4152542372881356
```

F1曲線180点のargmaxをそのまま転送する`oof_best`より自由度が低く、今回のfresh監査でも
閾値のrep間標準偏差を`0.04861 -> 0.01034`へ縮めた。

---

## 3. 全方面の施策棚卸し

既存成果と今回の再監査を、改善レバー別に整理する。

| 面 | 確認済みの結果 | 今回の判断 |
|---|---|---|
| 評価設計 | 自己資本比率の列衝突とBラベルearly-stopping漏洩を修正済み | strict方式以外を採否に使わない |
| 財務前処理 | H2-both4、続くH2+H7-both4は修正後も15repでACCEPT | 現行4列を維持 |
| 重複削減 | H52(E6重複/レア削除)、H54(E1会計重複削除)は改善なし | 列削減だけを目的に再開しない |
| 線形前処理 | raw金額列の歪度が極端。今回signed-logを検証 | Expert単体は改善、最終合成は不採用 |
| テキスト表現 | E4の組織図+企業概要embeddingは採用済み。SetFit/追加embedding/kNNは増分なし | E4を維持、新表現は新情報源がある場合のみ |
| モデル族 | CatBoost、RBF/GP、TabPFN、GAM、通常kNN等は最終合成を改善せず | 同じ入力上のモデル差し替え探索を止める |
| 交互作用 | E7は採用。H34収益×DX、H35人材軸は強い土台上でもREJECT | 生列の積を追加する探索を止める |
| Expert構成 | E7、E0bの追加は有効。H30/H31/H32再監査はREJECT | 現行8本を維持 |
| meta | 非負logit metaが等分より安定。高相関Expert追加は重みを奪い合う | 構成員追加には最終APの純増を必須とする |
| seed平均 | 5seed平均はOOF APを上げたがPublic F1 `0.8000 -> 0.77419` | 二値提出には使わない |
| 残差補正 | error検出はできても補正方向を予測できず、訂正器は悪化 | 同じ残差特徴で再開しない |
| 閾値 | 旧6則では`half_f1`が最良。今回fresh seedとPlatt版を追加 | 未校正`half_f1`を維持 |

現在の主な限界は、個別Expertを強くすることではなく、**既存Expertと異なる正しい残差を
安定して作れないこと**である。単体AP、低相関、meta重みのいずれも、最終増分の十分条件ではない。

---

## 4. 実験前監査

### 4.1 再現性

実行:

```bash
.venv/bin/python exp/repro_check.py
```

結果:

- Python `3.10.12`、主要依存パッケージは`requirements.txt`と完全一致。
- E3/E4の同一入力2回学習は最大差0。
- 参照exp026との最大確率差は`2.4e-15`。
- 参照提出ラベルとの差は0行。
- 総合判定: **OK**。

したがって、今回の値は過去記録と直接比較できる。

### 4.2 数値分布診断

E0bが見る数値93列のうち46列は6水準以上。特に金額・規模23列は裾が非常に重かった。

| 列 | raw歪度 | 符号付きlog後の絶対歪度 |
|---|---:|---:|
| 負債 | 23.34 | 0.78 |
| 固定資産 | 23.28 | 0.63 |
| 総資産 | 22.32 | 0.64 |
| 流動資産 | 20.78 | 0.62 |
| 事業所数 | 19.43 | 0.69 |
| 営業CF | 19.14 | 2.96 |

対象23列の絶対歪度中央値は`13.93 -> 1.17`へ下がる。この観察からH61を立てた。

---

## 5. H61: E0bの金額・規模列を符号付き対数へ変換

### 5.1 仮説

StandardScalerだけでは、極端値が標準偏差を支配して大多数の企業を狭い範囲へ押し込む。
金額・規模23列だけに次を適用すれば、符号を保ったまま線形Expertの境界が安定する。

\[
x' = \operatorname{sign}(x)\log(1+|x|)
\]

カテゴリ、比率、アンケート、辞書特徴、欠損フラグは変換しない。変換はtarget-freeであり、
foldごとのラベル統計を使わない。

### 5.2 実装

`exp/ensemble_experts.py`と`exp/holdout_check.py`へ、既定値を変えない
`--linear-preprocess {raw,signed_log}`を追加した。キャッシュ名も分け、raw結果の誤再利用を防ぐ。

### 5.3 スクリーニング

開発に使っていないseed 100〜104、strict B=260行 x 5repで比較した。

```bash
.venv/bin/python exp/holdout_check.py \
  --n-reps 5 --seed-offset 100 \
  --concat-embed --cross --e7-model lr \
  --linear --linear-c 0.03 --linear-preprocess signed_log \
  --h2h7-both4 \
  --threshold-rules oof_best half_f1 platt_half_f1 \
  --out-csv exp/holdout_h2h7both4_signedlog_thresholds_seed100_5rep.csv
```

#### E0b単体

| 指標 | raw | signed-log | ペア差 |
|---|---:|---:|---:|
| AUC | 0.9276 | 0.9388 | **+0.01120（5/5）** |
| AP | 0.8271 | 0.8487 | **+0.02160（4/5）** |
| F1 (`oof_best`) | 0.7267 | 0.7472 | **+0.02049（5/5）** |
| F1 (`half_f1`) | 0.7305 | 0.7530 | **+0.02248（5/5）** |

E0b単体では明確に成功した。

#### 8本合成でraw E0bをsigned-log E0bへ置換

| 指標 | raw構成 | signed-log構成 | ペア差 |
|---|---:|---:|---:|
| AUC | 0.9618 | 0.9613 | -0.00052（1/5） |
| AP | 0.8924 | 0.8919 | -0.00047（3/5） |
| F1 (`oof_best`) | 0.8051 | 0.8076 | +0.00247（2/5） |
| F1 (`half_f1`) | 0.8076 | 0.8191 | +0.01147（3/5） |

単体改善は最終順位へ伝わらなかった。主判定のAP/AUCを満たさないため、本命置換へは進めない。
ただし「log版がraw版と違う有効信号を持ち、置換ではmetaが利用できなかった」可能性が残った。
この結果からH62を立てた。

---

## 6. H62: raw E0bとsigned-log E0bを別Expertとして併存

### 6.1 仮説

rawの絶対スケール情報を残したまま、裾を圧縮したlog版を9本目として追加すれば、metaが行ごとに
補完的な信号を配分できる。

### 6.2 実装と評価

`--linear-log-extra`を追加し、同一rep内で8本構成と9本構成を比較した。seed 100〜104、
strict 5rep、他条件はH61と同じ。

```bash
.venv/bin/python exp/holdout_check.py \
  --n-reps 5 --seed-offset 100 \
  --concat-embed --cross --e7-model lr \
  --linear --linear-c 0.03 --linear-log-extra \
  --h2h7-both4 \
  --threshold-rules oof_best half_f1 platt_half_f1 \
  --out-csv exp/holdout_h2h7both4_dual_linear_thresholds_seed100_5rep.csv
```

| 指標 | 8本 | 9本（raw+log） | ペア差 |
|---|---:|---:|---:|
| AUC | 0.9618 | 0.9603 | **-0.0014（2/5）** |
| AP | 0.8924 | 0.8935 | +0.0011（2/5） |
| F1 (`oof_best`) | 0.8051 | 0.7956 | **-0.0095（2/5）** |
| F1 (`half_f1`) | 0.8076 | 0.8054 | -0.0022 |

log版の正規化meta重みは平均21.2%、raw版も13.8%残ったが、既存E2/E0/E4から重みを奪い、
F1ガードレールを破った。seed 42全OOFでraw/logのSpearmanは`0.97657`と非常に高く、
独立Expertと呼ぶには冗長だった。判定は**REJECT**。15repへ拡大しない。

H61/H62から得た一般則は、**前処理で単体モデルが強くなっても、高相関ならExpert追加の価値は
ほぼ無い**ということである。今後は単体改善ではなく、既存合成へのAP純増を最初のゲートにする。

---

## 7. H63: Platt校正後の`half_f1`

### 7.1 仮説

`half_f1`の理論的な根拠は確率校正を仮定する。AのOOF logit 1列だけにL2 Logistic Regression
（傾き+切片）をfitし、校正後の`F1*/2`を生スコア閾値へ逆変換すれば、閾値転送がさらに安定する。
Bラベルは校正器にも使わない。

### 7.2 fresh 15rep

seed 100〜104の5repと、105〜114の10repを事前固定した同一仕様で実行し、結合した。

```bash
.venv/bin/python exp/holdout_check.py \
  --n-reps 10 --seed-offset 105 \
  --concat-embed --cross --e7-model lr \
  --linear --linear-c 0.03 --h2h7-both4 \
  --threshold-rules oof_best half_f1 platt_half_f1 \
  --out-csv exp/holdout_h2h7both4_raw_thresholds_seed105_10rep.csv
```

meta_8本の結果:

| 閾値則 | B上F1 mean±std | 閾値 mean±std | B予測正例率 mean±std | vs `oof_best` |
|---|---:|---:|---:|---:|
| `oof_best` | 0.79328 ± 0.03112 | 0.37667 ± 0.04861 | 0.26179 ± 0.03472 | — |
| `half_f1` | 0.80155 ± 0.03670 | **0.39512 ± 0.01034** | 0.24513 ± 0.03259 | **+0.00826、10勝3同値2敗** |
| `platt_half_f1` | **0.80420 ± 0.03010** | 0.38041 ± 0.02123 | 0.25359 ± 0.02178 | **+0.01091、11勝3同値1敗** |

Platt版は`oof_best`より良い。しかし採否で比較すべき相手は、既に採用済みの`half_f1`である。

```text
platt_half_f1 - half_f1
平均 +0.00265 / 4勝4敗7同値 / 中央値 0.00000
```

差は4repだけ改善、4rep悪化、半数近く同値である。追加2パラメータに対する安定した純増とは
言えないため、Platt版は**REJECT**し、低自由度の`half_f1`を維持する。

### 7.3 開発seedからfresh seedへの再現

| 評価集合 | `half_f1 - oof_best` 平均ΔF1 | 改善/全rep |
|---|---:|---:|
| 開発seed 0〜14（bug修正済み） | +0.00947 | 12/15（同値1） |
| fresh seed 100〜114 | +0.00826 | 10/15（同値3） |

勝率は下がったが、平均改善量は非常に近い。`half_f1`の採用根拠は未知分割でも再現した。

---

## 8. 最終判断

### ACCEPT

- strict完全ホールドアウトを評価の唯一の正式経路とする。
- H2+H7-both4修正版の8 Expert構成を維持する。
- `half_f1`を最終閾値則として維持する。
- H61/H62/H63の実験オプションと結果CSVを再現用に残す。

### REJECT / 再開しない

- signed-log E0bへの本命置換: 単体は強いが合成AP/AUCの純増なし。
- raw/log E0b二重化: 高相関でF1悪化。
- Platt校正: `half_f1`への純増が4勝4敗7同値。
- Public結果を使った閾値、正例数、対象行の追加調整。
- 同じ入力に対するモデル族や高相関Expertの追加。

### 条件付きで残す研究候補

signed-log E0bは単体改善が明確なので、将来Expert数を減らす必要が生じた場合の代替候補にはなる。
ただし今回の結果を見た後で「変換列を選ぶ」「raw/logを手動配分する」追試はpost-hoc探索になる。
再開するならseed 200以降を一度だけ使い、仕様を先に固定し、最終合成ΔAP `>= +0.005`を必須にする。
現時点では期待値より多重検定リスクが高いため、実行を推奨しない。

---

## 9. 実装・成果物

### 変更コード

- `exp/ensemble_experts.py`
  - `--linear-preprocess {raw,signed_log}`
  - `--linear-log-extra`
  - 変換別キャッシュタグ
- `exp/holdout_check.py`
  - 上記2オプションのstrict holdout配線
  - `platt_half_f1`
  - A/B表示を実行実数の482/260へ修正

既定値は`linear_preprocess=raw`、`linear_log_extra=False`であり、従来予測を変えない。

### 新規結果

- `exp/holdout_h2h7both4_raw_thresholds_seed100_5rep.csv`
- `exp/holdout_h2h7both4_raw_thresholds_seed105_10rep.csv`
- `exp/holdout_h2h7both4_signedlog_thresholds_seed100_5rep.csv`
- `exp/holdout_h2h7both4_dual_linear_thresholds_seed100_5rep.csv`

### 回帰確認

- `python -m py_compile exp/ensemble_experts.py exp/holdout_check.py`: OK
- `git diff --check`: OK
- 既定`fit_e0b()`と`numeric_transform="raw"`の予測完全一致: OK
- `exp/repro_check.py`フル: OK
- 参照提出ラベル差: 0行

新しい提出CSVは作っていない。最終提出候補は第0節の既存ファイルのままである。
