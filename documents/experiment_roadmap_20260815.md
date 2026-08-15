# SIGNATE Student Cup 2026 実験ロードマップ

作成日: 2026-08-15  
提出締切: **2026-09-04 00:00**  
基準モデル: **exp032**（Public F1 0.80000 / OOF F1 0.8061 / OOF AUC 0.9573）

## 1. 目的

締切までに狙うのは、既存特徴の細かな再加工ではなく、exp032がまだ持っていない
**新しい表現・距離・一般化バイアス**を持つ候補を1本以上作ることである。

探索の優先順位は、単体性能ではなく次で決める。

\[
\text{実験価値}
\approx
\text{exp032への増分性能}
\times
\text{残差の独自性}
\times
\text{seed間安定性}
\div
\text{検証コスト}
\]

現時点での本命仮説は、購入ラベルを使って文章表現自体を更新する
**target-aware text representation** である。

## 2. 今回は追わないもの

以下は既存実験で十分に否定されたか、締切までの期待値が低いため原則再開しない。

- 財務比率、survey変換、通常のtarget encodingの追加
- LightGBM/CatBoost/Optunaなど、既存座標上の単体モデル強化
- 生スコアに対するthreshold ruleの追加探索
- 固定汎用embedding上の通常kNN
- product特徴を大量生成するAutoCross型探索
  - E7の事後分解では、積12列の純増ではなくNeed/Capacity親軸の独立Expert化が成功要因だった
- pseudo-label、train+test事前学習、synthetic augmentation
  - 現行大会ルールの確認前は着手しない
- cross-attention、TabR、TabMなどの高コスト手法を無条件に始めること

## 3. 全実験共通の判定フロー

### Stage 0: 合法性・再現性・リーク確認

実装前に次を満たす。

- 外部pretrained model、外部データ、test利用に関する現行大会ルールを確認し、根拠を記録する
- `python3 exp/repro_check.py` がOKである
- `random_state`、データ行順、企業ID、欠損処理を固定する
- 学習済み重み・生成embedding・依存パッケージのバージョンを記録する
- validation labelがencoder、特徴選択、近傍DB、calibrationへ入らないことをコード上で確認する

Stage 0を通らない候補のスコアは比較表へ入れない。

### Stage 1: Discoveryスクリーニング

固定済みdiscovery 482件だけを使い、最小構成を評価する。最初は1 seedで配管を確認し、
問題がなければ3〜5 seedへ進む。ここではハイパーパラメータを広く探索しない。

見るもの:

- 候補単体のAUC / AP / F1
- 既存Expertおよびexp032とのSpearman相関
- exp032のFN rescue、FP追加、error overlap
- exp032へ追加・置換したときのpaired ΔAUC / ΔAP / ΔF1
- メタ重みとseed間のばらつき

早期停止:

- exp032追加時の平均ΔAPが負
- F1が一貫して悪化
- 単体信号が正例率とほぼ同等
- 高相関かつ追加ΔAPがほぼゼロ
- candidate重みがほぼゼロで、固定alpha対照でも改善しない

相関の低さだけでは昇格させない。exp035は最大相関0.824でも合成後に負けている。

### Stage 2: 仕様固定と安定性確認

Stage 1を通った候補だけ、モデル・入力列・前処理・主要ハイパーパラメータを固定し、
discovery内の10〜15 seed paired comparisonへ進める。

スクリーニング基準は現行 `exp/decision.py` に合わせる。

| 指標 | 昇格基準 |
|---|---|
| AP | 平均ΔAP ≥ +0.005、15回なら12/15以上で改善 |
| AUC | 15回なら11/15以上で改善 |
| F1 | 平均ΔF1 ≥ -0.002を最低条件とし、最終提出候補は平均ΔF1 > 0を要求 |
| Incremental value | 候補単体ではなく `exp032 + candidate` が改善 |
| Meta stability | 重みがほぼゼロ、またはfoldごとに消える場合は固定alpha対照を追加 |

### Stage 3: Lockbox一回評価

仕様固定後にだけ、固定lockbox 260件を一度評価する。

- discoveryだけでencoder、Expert、meta、alpha、thresholdを学習する
- lockboxのlabelは予測生成後まで参照しない
- exp032と候補を同じlockbox上でpaired比較する
- lockboxを見てモデル、列、epoch、thresholdを変更しない

通過候補だけ全742件で最終学習し、提出候補を作る。lockbox不通過なら、理由を記録して終了する。

## 4. 優先順位

| 順位 | 実験 | 目的 | 工数目安 | 実施条件 |
|---:|---|---|---:|---|
| 0 | ルール確認・exp032凍結 | 外部モデル利用可否と比較基準を固定 | 0.5日 | 最初に必須 |
| 1 | Target-aware text / SetFit | 購入タスク専用の文章座標を作る | 3〜4日 | ルール通過 |
| 2 | Structured-only TabPFN | 木・LRと異なるtabular priorを得る | 1〜2日 | pretrained model利用可 |
| 3 | Residual predictability診断 | exp032の誤りがそもそも予測可能か判定 | 1日 | 既存clean OOFを使用 |
| 4 | Structured-only ModernNCA | 購入ラベル用の近傍空間を学ぶ | 2〜3日 | 上位実験と計算資源が競合しない |
| 5 | Conditional residual correction | 新表現でexp032だけを補正 | 1〜2日 | residual診断通過時のみ |
| 6 | Nested GES | 新旧Expertの再選択・再配分 | 1日 | 新候補が2本以上揃った後 |
| 7 | Target-aware text拡張 | DX展望以外、複数文書へ拡張 | 1〜2日 | SetFit最小構成が通過 |
| 8 | TabM / optimal binning / calibration | 予備候補 | 各1〜2日 | 上位候補が全滅し、8/28以前のみ |
| HOLD | cross-attention / TabR / AutoCross / pseudo-label等 | 高コストまたは根拠不足 | — | 原則実施しない |

H41のローカルBERT埋め込み比較は、現在すでに実装途中なので1日以内で完了させる。
ただしこれはfrozen encoderの差し替えであり、順位1のtarget-aware実験とは別仮説として扱う。
H41が不発でもSetFit仮説は棄却しない。

## 5. 実験カード

### R1. Target-aware text representation

**仮説**  
TF-IDFやOpenAI embeddingでは固定されていた文章間距離を購入ラベルで更新すると、
exp032が外す企業を救う新しい順位が得られる。

**最小構成**

- 入力: `今後のDX展望` だけ
- encoder: 規約上利用可能な日本語または多言語SentenceTransformerを1種類
- objective: SetFitまたはsupervised contrastive + classification
- head: Logistic Regressionまたは小さなlinear head
- 候補の出力: 1本のExpert確率
- 比較: E3 TF-IDF、E4 frozen embedding、exp032

**必須のCV構造**

外側foldごとに次を最初からやり直す。

`outer train label → pair生成 → encoder fine-tune → head学習 → outer validation予測`

全742ラベルでencoderをfine-tuneしてからCVすることは禁止する。pair数が増えても独立企業数は
742のままなので、「pairを増やせば小標本問題が解消する」とは解釈しない。

**昇格条件**

- E3より単体APが改善する、またはE3と異なるFNを安定して救う
- exp032追加時にStage 2のΔAP/F1条件を満たす
- seedを変えても相関・重み・FN rescueが大きく崩れない

**停止条件**

- exp032との順位相関が高く、追加ΔAP < +0.002
- fine-tune seedで性能・予測順位が大きく反転
- 単体改善があってもexp032内でE3/E4の役割を移すだけ

**通過後の拡張は一つだけ選ぶ**

1. 企業概要 + 組織図へ同じ方式を適用
2. E3を置換
3. exp032へ9本目として追加

複数列融合、ArcFace、cross-attentionを同時に始めない。

### R2. Structured-only TabPFN

**仮説**  
事前学習されたtabular priorが、E0のtree partitionとE0bのlinear boundaryにない残差を持つ。

**最小構成**

- text本文、TF-IDF、embedding全次元は入れない
- 最初は構造化特徴だけ
- 大規模なハイパーパラメータ探索はしない
- 第2段階へ進んだ場合だけE3/E4のcross-fitted scoreを各1列追加する
- 比較: E0、E0b、exp032

**停止条件**

- E0/E0bとほぼ同じ順位でexp032追加ΔAP < +0.002
- CPU/GPU/メモリ要件が想定を超え、8/21までにclean OOFが作れない
- pretrained modelの利用条件を確認できない

### R3. Residual predictability診断

**仮説**  
exp032の誤りに再現可能な構造があるなら、新表現またはExpert間不一致から補正方向を予測できる。

**入力候補**

- 完全cross-fitted `p_base`
- Expert間の標準偏差とmax-min
- E3とE4、E0とE7などの予測差
- R1/R4が作れた場合はtarget-aware scoreまたはlocal density

**評価**

- `error = 1[pred_base != y]` のAUC
- `residual = y - p_base` の符号・大きさの予測
- shuffled target対照

base OOFを作った同じ行で補正器を学習・評価してはいけない。補正器にももう一段の
cross-fittingを入れる。

**分岐条件**

- error AUCが複数splitで安定して0.60〜0.62を超える: R5へ進む
- 0.5付近、またはshuffle対照との差が不安定: residual路線全体を終了

### R4. Structured-only ModernNCA

**仮説**  
固定cosine-kNNでは弱かった近傍信号を、購入ラベル用に距離関数から学び直すことで
実用的な性能まで高められる。

**最小構成**

- structured featuresのみ
- numericは標準化 + missing indicator
- categoricalはone-hotまたは低次元embeddingのどちらか一方
- validation行とvalidation labelを近傍DBへ入れない
- class-balanced batch/sampling
- 比較: exp035固定kNN、E0、E0b、exp032

**停止条件**

- exp035より単体性能が明確に上がらない
- seed間で近傍構造と予測順位が大きく崩れる
- diversityがあってもexp032追加時にΔAP/F1が出ない

### R5. Cross-fitted residual correction

R3通過時だけ実施する。

\[
\operatorname{logit}(p_{new})
=
\operatorname{logit}(p_{base}) + g(z)
\]

- `g` はridgeまたは低自由度logistic correction
- 既存94列を全部渡さず、不一致・新表現・局所密度に限定する
- base prediction、補正器、thresholdをすべて外側train内で学ぶ
- exp032への独立Expert追加と直接比較し、correctionの方が良い場合だけ残す

### R6. Nested Caruana GES

新しい候補予測が2本以上揃ってから実施する。

候補ライブラリ:

- exp029 / exp031 / exp032
- E0〜E7 / E0bの個別予測
- R1 SetFit
- R2 TabPFN
- R4 ModernNCA
- H33 multi-seed scoreは順位素材としてのみ候補に含める

APをgreedy objective、F1をguardrailとする。ただし選択と評価を同じOOF上で行わず、
外側train内だけで構成員と反復回数を決め、外側validationで評価する。

GESがexp032に勝っても、選ばれた構成員がsplitごとに全面的に入れ替わる場合は採用しない。

## 6. 日程

### 8/15〜8/16: 基準凍結と入口整理

- [ ] 現行大会ルールで外部pretrained model利用可否を確認・記録
  - 公開検索では規約本文を取得できなかった。提出昇格前に大会画面での確認が必要
- [x] exp032の再現確認と提出ファイルhashの記録
- [ ] H41を1日で完了または停止
- [ ] R1/R2/R4用の依存パッケージを通常環境から分離
- [ ] 各新モデルが出すOOF/test確率の共通インターフェースを決める

成果物: ルール根拠、再現ログ、H41判定、実験テンプレート。

### 8/17〜8/21: 最優先スクリーニング

- [x] R1 SetFit-styleの1 seed smoke → discovery 5-fold
  - 最小構成はREJECT。単体AP 0.4452、exp032追加ΔAP -0.0022 / ΔF1 -0.0031
  - lockboxは未開封。設定探索や追加fine-tuneは行わず、R2へ進む
- [ ] R2 structured-only TabPFNのdiscovery評価
  - package導入と実装は完了。公式weight取得にPrior Labsのライセンス承諾と
    `TABPFN_TOKEN`が必要なためsmoke前で保留
- [x] R3 residual predictabilityを既存OOFで診断
  - error AUC 0.8399、confidence-only比 +0.0929で診断gateは通過
  - 条件付きR5はΔAP -0.0155 / ΔF1 -0.0044でREJECT。correction路線は終了
- [x] AutoCross-inspired 2-way cross探索 + sparse LR
  - fold内quantile bin、325候補、inner 3-fold screening + greedyで完全nested評価
  - 単体AP 0.6228、exp032追加ΔAP -0.0183 / ΔF1 -0.0115、cross選択の
    fold間Jaccard 0.098。inner改善がouterへ再現せずREJECT
- [ ] 候補ごとに相関、error overlap、FN rescue、meta weightを保存

8/21終了時点で、各候補を `昇格 / 保留 / 終了` の3つに分類する。

### 8/22〜8/26: 安定性確認と条件付き第二陣

- [ ] R1/R2の通過候補を10〜15 seedへ拡張
- [x] R4 ModernNCAを最小構成で評価
  - TALENT公式構造・既定値をstructured-only 47列へ適用。単体AP 0.3942で
    exp035固定kNN 0.5847を下回り、exp032追加もΔAP -0.0115 / ΔF1 +0.0081
  - AP主目的と「exp035を明確に上回る」の両条件に失敗したためREJECT。
    lockbox・seed展開・epoch/temperature探索へは進まない
  - [x] H43 exp032を80%以上固定したanchored R4を追加診断
    - nested λ/threshold転送でΔAP +0.0046まで回復したが、転送F1は+0.0011。
      事前条件+0.005未達かつ1fold大崩れのためREJECT
    - ユーザー判断でlambda=0.07のchallengerを1点提出。Public 0.776119で、
      exp032から追加したPublic 4件が全FP。再調整せず終了、本命0.80000を維持
  - [x] H44 陽性数制約付きrank blend
    - exp032転送thresholdの陽性数Kを固定し、rank blend上位Kへ同数交換。
      NestedでΔAP +0.0149 / Δ同一K F1 +0.0079、悪化foldなし、誤り62→60。
    - lambda選択は0.15 / 0.15 / 0.10 / 0.10 / 0.10。平均0.12を固定した
      K=236 challengerを作成し、testでは各方向10件ずつ交換。本命は維持
    - Public実測 0.781955（2026-08-16）。復元は TP=52 / FP=22 / FN=7 で
      **TPは不変、FPのみ+3**。test全体では10/10の対称交換でも、Public 240行では
      `0→1`が3件多く落ちた＝陽性数の固定は部分集合では非対称に崩れる。
      Public差 -0.0181 は実測std 0.043 の内側なので単独ではREJECT根拠にならず、
      Private自動採用のため保持コストもゼロ。**第2枠として保持**し、
      lambda/K/閾値のPublic基準の再調整は行わない
  - [x] H45 境界限定R4/R1双方向swap
    - exp032のthreshold境界だけを固定Kで交換したが、ΔAP -0.0019 / Δ固定K F1
      -0.0472、fold 1/5で崩壊。R1のboundary selector利用はREJECT、提出なし
  - [x] H46 H44 R4 rank blendの交換数cap
    - APは+0.0105まで改善したが、固定K F1差0.0000、fold 5で-0.0417。
      F1条件未達のためREJECT、提出なし。H44をローカル最良として維持
- [x] R3通過時のみR5 correctionを実施
  - 実施済み。ΔAP -0.0155 / ΔF1 -0.0044でREJECT
- [ ] SetFit通過時のみ、企業概要+組織図への拡張を一つ試す

8/26以降、新しい高コストモデルには着手しない。

### 8/27〜8/30: 最終合成

- [ ] 全通過候補を同一splitのclean予測へ揃える
- [ ] exp032への単純追加・置換を正式比較
- [x] 新候補が2本以上ある場合だけR6 nested GES
  - 15予測をAP-greedy、exp032 train-F1以上をguardrailとしてouter 5-fold評価
  - train APは全foldで改善したがvalidationは4/5悪化。全体ΔAP -0.0146 /
    ΔF1 -0.0199でREJECT、lockbox・提出へは進まない
- [ ] 仕様固定後、lockboxを一度だけ評価
- [ ] 最終候補の予測正例率、境界行、seed分散を点検

8/30終了時点でモデル仕様とthreshold決定方法を凍結する。

### 8/31〜9/03: 提出・再現・説明資料

- [ ] 全742件で最終学習し、提出CSVを生成
- [ ] CSVの件数、企業ID、ラベル値、正例数、hashを確認
- [ ] exp032との差分行数と `0→1 / 1→0` を記録
- [ ] 最終2枠を選ぶ
- [ ] `repro_check.py` と最終生成コマンドを保存
- [ ] 重要Expert、購入企業像、施策提案をプレゼン用に整理

提出操作は**9/03 21:00 JSTまで**を内部締切とし、公式締切直前の再学習や仕様変更を禁止する。

## 7. 最終提出2枠の決め方

### 枠1

**exp032を必ず残す。** Public 0.80000かつ完全ホールドアウトで確認済みの基準であり、
新候補のPublic値を見て差し替えない。

### 枠2

次をすべて満たす最良のchallengerを置く。

- exp032追加または置換でStage 2を通過
- lockboxでF1ガードレールを破らない
- exp032と十分な差分行を持ち、Privateに対するヘッジになる
- 再現コマンドとモデル資産が揃っている

challengerが一つも通らなければ、無理に新規モデルを提出せず、exp031など既存の安定構成を
第2枠候補として再比較する。Publicスコアだけで第2枠を選ばない。

## 8. 実験ごとに残す記録

各実験は最低限、次を `documents/experiments.md` または `exp/hypothesis_log.md` に残す。

- 仮説と既存Expertとの差
- 使用列、モデル、重み、依存バージョン
- CV分割、seed、fold内学習範囲
- 単体指標とexp032への増分指標
- Spearman、error overlap、FN rescue / FP追加
- meta weightの平均・標準偏差
- ACCEPT / PARK / REJECTと停止理由
- 実行コマンド、生成ファイル、所要時間
- ルール上の根拠が必要な場合はその参照先

## 9. 成功条件

このロードマップの成功は、試したモデル数では測らない。

- target-aware representationが有効か、リークなしで結論を出す
- 少なくともS/A候補3系統を低コストで判定する
- exp032を壊さず、通過したchallengerだけを第2提出枠へ進める
- 全候補が不発でも、8/30までに探索を止めて再現性と説明資料を完成させる

最終的に新候補が採用されなくても、「購入ラベル用の意味空間・tabular prior・教師あり近傍空間」
の三つを正しく検証し、exp032がそれらを含めても残る基準だと確認できれば、探索としては完了とする。
