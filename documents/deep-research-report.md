# SIGNATE Student Cup 2026 停滞突破のための類似コンペ徹底調査

## 現在の戦略と停滞の正体

まず結論から言うと、**現状は「モデルが弱い」のではなく、「同じ情報を別の形で再学習する実験が限界に達している」状態**です。GitHub の実験履歴を追うと、この診断はかなり明確です。

train は742社、正例179社で正例率24.12%、test は800社です。再点検では構造化特徴・各テキスト埋め込みを使った adversarial validation がほぼ AUC 0.5 に留まり、train/test の強い分布シフトは確認されていません。したがって、現時点では「CVの分割が根本的に間違っている」「testだけ別母集団」という説明は弱いです。fileciteturn4file0

### 現在の実質的な best 構成

現行の基準は **exp032** とみなすのが合理的です。Public F1 は **0.80000**、OOF F1 0.8061、OOF AUC 0.9573。さらに完全ホールドアウトでは AUC 0.9547、AP 0.8830、F1 0.7891 と、単発OOFだけでなく再現性検証でも強い結果になっています。exp032 は exp031 に E0b linear expert を加え、15 repeated holdout で ΔAP +0.0070、12/15勝、ΔF1 +0.0107、11/15勝で ACCEPT されています。fileciteturn2file0

現在の Expert 構成を整理すると次の通りです。

| Expert | 入力・モデル | 現在の役割 |
|---|---|---|
| E0 anchor | 約94列＋nested text score → LightGBM | 全体の強いアンカー |
| E1 finance | 財務・企業規模 → LightGBM | 財務固有信号 |
| E2 survey | アンケート12列 → Logistic Regression | 木と異なる低次元線形信号 |
| E3 DX text | DX展望 → TF-IDF + LR | 語彙ベースの将来意欲 |
| E4 text | 組織図＋企業概要の OpenAI embedding 連結 → LR | 汎用意味空間による文章信号 |
| E6 manual | 辞書・構造特徴 → LightGBM | 手作業意味特徴 |
| E7 cross | 不満 × 財務の12交互作用 → LR | Need × Capacity 型の明示交互作用 |
| E0b linear | E0と同じ入力 → one-hot + StandardScaler + LR | 木と違う決定境界 |

E0〜E6 の特徴量グループ別 Expert 化そのものが exp018 で大きな改善を生み、その後 E4 の frozen embedding 化、E7 の交互作用 LR、E0b の線形モデルが段階的に上乗せされています。メタ層は各 Expert の logit を非負重みで結合する logistic meta learner で、alpha は学習側内部CVの AP で選択されています。fileciteturn9file0 fileciteturn10file0

### 何が本当に効いたか

履歴の中で重要なのは、**「単体モデルを強くした実験」ではなく「既存モデルと違う情報・表現を持ち込んだ実験」が最終アンサンブルを押し上げている**ことです。

典型例は E4 の変更です。組織図 TF-IDF を単に別モデルで学習するのではなく、組織図＋企業概要の frozen embedding 連結へ**置換**した exp029 は、15 holdout で exp019 比 ΔAP +0.0072、13/15勝、ΔF1 +0.0087、11/15勝でした。E4 単体でも ΔAP +0.0378、10/10勝、既存最大相関0.878と、性能と diversity を両立しています。fileciteturn2file0

さらに E7 の「不満 × 財務」12交互作用を LR Expert として追加した exp030 は、15 holdout で ΔAP +0.0136、14/15勝、ΔF1 +0.0115、10/15勝。これは単なる財務比率追加よりはるかに大きく、メタ重みも21.5%を得ました。fileciteturn2file0

exp032 の E0b も重要です。同じ約94列を使うにもかかわらず、木ではなく one-hot + LR にするだけで exp031 から ΔAP +0.0070、ΔF1 +0.0107。**新しい原データを追加していないのに、表現基底・決定境界を変えることで残差信号を得ています。** fileciteturn2file0

これは今後の探索方針を決める非常に強い証拠です。

### 何が効かなかったか

一方、単体Expertの改善がアンサンブル改善につながらない例が繰り返されています。

exp023 の財務比率は E1 単体では明確に改善したにもかかわらず、最終ensembleでは exp019 比 F1 -0.0092、1/5勝で REJECT。exp025 のアンケート段階特徴も E2 単体では ΔAP +0.0151、20/20勝なのに、合成後は ΔAP -0.0051、1/5勝でした。exp034 の GAM は E0b より単体APが高かったにもかかわらず、exp032へ入れると ΔAP -0.0005、ΔF1 -0.0058。exp037 の新しい財務×DX交互作用も、見かけ上の微改善はメタ alpha 選択が変わった副作用で、候補自身の重みは平均0.4%しかなく停止されています。fileciteturn2file0

Survey の RBF-SVM、GP、ARD-GP も、モデル族を変えたにもかかわらず既存順位と r=0.916〜0.983に収束し、全て REJECT でした。つまり、**同じ12列から抽出できる順位情報そのものがほぼ飽和している**と読めます。fileciteturn2file0

手作業特徴でも、財務健全性、従業員あたり指標、ニーズギャップ、通常の業界 target encoding、キャッシュフロー比率などは既に検証・棄却されています。特に通常TEは CV 内 fit まで実装した上で有害でした。fileciteturn5file0

したがって、今から「別の財務比率」「別のsurvey変換」「別のGBDT」「LightGBMのOptuna」を掘る期待値は低いです。

### CV・threshold 設計について

このチームの評価設計は、一般的なコンペ参加者よりむしろ厳格です。単seed F1は分散が大きく、乱数特徴でも +0.013程度出ることを実測し、AUC＝検出、AP＝F1に繋がる順位性能、F1＝guardrail として複数seedの paired comparison を使っています。Notebook 系でも ACCEPT は原則「平均 ΔAP ≥0.005 かつ70%以上の勝率」です。fileciteturn5file0 fileciteturn6file0

さらに `holdout_check.py` は通常の stacking OOF に残る restacking optimism を明示的に認識し、A/B完全分離後、A内だけでExpert OOF・meta・alpha・thresholdを決め、A全体再学習後に B を評価しています。これは今後も維持すべきです。fileciteturn11file0

thresholdについても `oof_best`, fold mean, rate matching, calibrated-F1由来の half-F1 型を既に比較し、現行 `oof_best` が維持されています。exp028 では threshold 周辺だけに明確な「誤りの塊」も発見できませんでした。fileciteturn2file0

つまり、**停滞原因を「閾値探索不足」と考えるのはかなり弱い**です。

### 現在の最大ボトルネック

私の診断は次です。

> **現在のモデル群は「財務・アンケート・語彙・汎用意味embeddingから取り出せる情報」はかなり回収済みだが、購入ラベル自身を使って“表現空間を変形する”探索をほとんどしていない。**

TF-IDF はラベルとは独立に語彙空間を作ります。OpenAI embedding も購入ラベルとは無関係な汎用意味空間です。LLM 3軸も、人間が決めた意味軸への投影です。その後に LR/LGBM が購入ラベルを学習しているだけで、**文章同士の距離そのものは購入タスクに適応していません**。fileciteturn2file0 fileciteturn9file0

これは、後述する類似コンペ・2024〜2026年の multimodal/tabular 研究と照らすと、現在の探索空間で最も大きな穴です。

## 類似コンペから見える勝ち筋

テーマではなく、**「異種データ」「不均衡」「二段階」「小〜中規模でのdiversity」「text representation」**という構造で比較しました。

| Competition | Data | Metric / task | 実際の上位解法で重要だったもの | 今回との共通点 | 参考度 |
|---|---|---|---|---|---|
| **Home Credit Default Risk** | 顧客・信用・過去契約など複数テーブル | binary / AUC | 集約特徴、子テーブルごとのモデル予測を上位特徴化、GBDT＋NN等のstack。Silver解法では Featuretools、GMM/KMeans 等も利用。citeturn10search0turn10search1 | 異種企業/顧客属性、class imbalance、expert prediction feature | ★★★★★ |
| **Avito Demand Prediction** | tabular＋タイトル＋説明文＋画像meta | regression | Gold上位解法は TF-IDFだけでなく、**文章から他のtabular属性を予測するmulti-task/self-supervised NN**や多層stackingを使用。citeturn3search14turn3search2 | text＋tabularの本格融合、表現学習 | ★★★★★ |
| **Mercari Price Suggestion** | 商品説明＋カテゴリ等tabular | regression | multimodal Transformer と tabular learner のstackが強力で、AutoGluon multimodal benchmarkでも実戦競争力が報告されています。citeturn7academia51 | 自由記述＋カテゴリ＋数値 | ★★★★☆ |
| **Santander Customer Transaction Prediction** | anonymized tabular | binary / AUC | 1st-place解法は pseudo-labeling の代表的事例として後続資料でも引用されています。citeturn12view0turn13search9 | binary、不均衡、rank性能重視 | ★★★☆☆ |
| **Porto Seguro Safe Driver** | 数値＋カテゴリ | binary / normalized Gini | NN entity embedding / denoising representation がtree ensembleと異なる信号源として上位stackに寄与した事例があります。citeturn9search0turn10search3 | mixed tabular、順位性能、ensemble diversity | ★★★★☆ |
| **Foursquare Location Matching** | 店舗名・住所などtext＋緯度経度等numeric | matching / ranking | 2022年3位は **XLM-R + ArcFace による教師ありmetric learning**で同一POIを近接させ、その後TP/FP classifierを入れる二段階構成。citeturn20search5 | text＋numeric、ラベルで表現空間そのものを学習 | ★★★★★ |
| **Feedback Prize – English Language Learning** | 比較的小規模な文章 | multi-target regression | 上位はfine-tuned Transformer＋seed ensemble。pseudo-labelは改善する場合とLB悪化する場合の両方が報告されています。citeturn8search9turn8search15 | small text、CV分散、pseudo-label risk | ★★★★☆ |
| **Child Mind Institute Sleep State** | 時系列＋event/meta | event detection / AP系 | 2位解法は stage-1 candidate prediction を stage-2 LightGBM で再スコアし、CVをさらに改善。citeturn20search2 | base scoreを第二モデルが補正するresidual/reranking | ★★★★☆ |
| **OTTO Recommender** | 行動session | Recall/MRR | candidate generation → XGB/CatBoost の二段階rerank → rank blend。citeturn9search11 | 「確率推定」より順位を再構成する発想 | ★★★☆☆ |
| **LEAP – Atmospheric Physics** | 大規模tabular regression | regression | 上位解法では best model の誤りからhard examplesを抽出し、他モデル学習へ重点投入する residual/hard-example 戦略が利用されました。citeturn20search3 | error-driven second modelという考え方 | ★★★☆☆ |
| **Happywhale** | image retrieval | MAP@5 | 1位は ArcFace系 metric learning＋kNN＋複数round pseudo-label。citeturn9search2 | 教師あり距離学習・retrieval・SSLの成功例 | ★★★☆☆ |
| **Tradeshift Text Classification** | document text | multilabel classification | 1st-place solution/code が公開され、text representation＋ensembleの典型的競技例。citeturn12view4turn13search6 | text分類、F系metricに近い | ★★★☆☆ |
| **Semi-Supervised Feature Learning** | labeled＋unlabeled | supervised downstream | 興味深いことに、semi-supervisedを主題とした競技でも winner は pure supervised baseline を採用しており、「unlabeled dataがある＝SSL有利」ではありません。citeturn9search3 | testを使う表現学習への警告 | ★★★☆☆ |

この比較で最も重要なのは、上位解法が必ずしも「より強い単体モデル」を探していないことです。

Avito や Foursquare では、**入力の表現そのものを学習対象にする**ことで既存GBDT/TF-IDFと異なる座標系を作っています。Child Mind や OTTO は、第一モデルの出力を最終答えとせず**第二段階で再順位付け**します。Home Credit 型は、異なるデータ源ごとに予測器を作り、その予測自体を別の特徴として利用します。citeturn3search2turn20search5turn20search2turn10search1

これは現在のリポジトリで成功した exp018→029→030→032 の進み方とも一致します。つまり、**新しい情報源、新しい表現、新しい予測原理をExpertとして入れた時だけ、アンサンブルが大きく伸びやすい**ということです。fileciteturn2file0

なお、公開検索では今回の「企業データからDX教育商材購入を予測する Student Cup 2026」の参加者向け Rule ページを確認できませんでした。検索で確認できた `Student Cup 2025` は2026年1〜2月開催の別のNLP課題で、今回のコンペ規約を代理できません。したがって **train+test joint learning、pseudo-label、外部pretrained modelの可否は、現在参加中コンペの画面上の規約を基準にする必要があります**。古いSIGNATE大会のルールから合法性を推測するべきではありません。citeturn23search1turn23search2

## 未実施アプローチと優先順位

以下は GitHub 上で確認できた実験を除外した上での候補です。「CatBoostをもう一度」「別の財務比」「Embeddingを別モデルに入れる」のような実質的重複は外しています。

| 手法 | 元になったコンペ / 論文 | 既存手法との差 | なぜ効く可能性があるか | Diversity期待 | 実装難易度 | 計算量 | Leakage / Rule Risk | 優先 |
|---|---|---|---|---|---|---|---|---|
| **教師ありSentenceTransformer / SetFit** | SetFit 2022、MulTaBench 2026。citeturn16search0turn16search8turn16academia48turn16academia49 | frozen embedding後にLRではなく、**targetでembedding空間自体を更新** | 現在未回収の「購入タスクに特有な意味距離」を作れる | **非常に高** | 中 | 中 | fold外labelをencoder fine-tuneに混ぜると致命的 | **S** |
| **TabPFN v2** | Nature 2025。small/medium tabular向け。citeturn14search0turn14search11 | GBDT/LRではなく事前学習済み tabular foundation model | 742行・数十〜100特徴は想定領域に近い | **高** | 低〜中 | 中 | pretrained weight の競技規約確認 | **S** |
| **AutoCross-inspired cross discovery + sparse LR** | KDD AutoCross。citeturn22search0 | 手作業12交互作用ではなく、探索的に高次crossを発見 | E7とE0bが両方成功しており、今回のデータに非常に強い事前証拠 | **高** | 中 | 低〜中 | cross選択をfold外で行うとfeature-selection leakage | **S** |
| **ModernNCA** | ICLR 2025。300 datasetsでCatBoost級の性能を報告。citeturn15search6 | exp035の固定cosine-kNNと違い、**ラベルで距離関数を学習** | 「誰が近い企業か」を購入ラベル用に再定義できる | **非常に高** | 中 | 中 | validation自身をneighbor poolに入れない | **S** |
| **Caruana Greedy Ensemble Selection** | ICML 2004、後続post-hoc ensemble研究。citeturn18search0turn18academia45 | 全モデルへ連続weightをfitする現metaと違い、incremental metricで構成員をgreedy選択 | 強いが冗長なGAM/E1等を落とし、弱いが直交したモデルを拾える | 高 | **低** | 低 | selectionと評価を同一OOFですると過学習 | **A** |
| **Tabular-Text Transformer / cross-attention** | ACL Findings 2024 TTT。citeturn17search0turn17search1 | text→probability後のlate fusionではなくintermediate fusion | 「財務状態によって同じ文章の意味が変わる」相互作用を学習可能 | **非常に高** | 高 | 高 | n=742でoverfit大 | **A** |
| **Cross-fitted residual logit correction** | Child Mind 2nd、LEAP hard-example。citeturn20search2turn20search3 | 独立Expert平均ではなく `base score + correction` | exp032が系統的に外す部分だけに容量を割ける | 高 | 中 | 低〜中 | base予測を必ずOOF化 | **A** |
| **TabM / BatchEnsemble MLP** | ICLR 2025 TabM。citeturn14search3 | 木・通常MLPと異なるparameter-efficient ensemble | 小規模tabularでNNの分散を抑えつつ非線形性を得る | 中〜高 | 中 | 中 | 低 | **A** |
| **TabR learned retrieval** | ICLR 2024。citeturn15search0 | fixed kNNでなく、representation＋retrievalをend-to-end学習 | 類似企業のラベルをcontextとして使える | 高 | 高 | 中〜高 | fold separation必須 | **A** |
| **Optimal binning + WOE + LR** | OptBinning。citeturn22academia48 | 木任せでなく、教師ありbinで低自由度の非線形linear expertを作る | E0bが効いたため、非線形basisをLRへ与える価値あり | 中 | 低〜中 | 低 | bin boundaryはfold内fit | **A/B** |
| **Calibration後のF1 decision rule** | F1最適decision theory。citeturn21academia48turn22search13 | 現exp024の生score threshold探索と違い、まず確率校正 | threshold varianceを少し削れる可能性 | 低 | 低 | 低 | nested calibration必須 | **B** |
| **Hierarchical / Bayesian interaction TE** | ordered target statisticsの原理はCatBoost。citeturn22search5 | H5の単純業界TEより、親カテゴリへshrinking | industry×size等の局所rateを線形Expertへ渡せる | 中 | 中 | 低 | **非常に高**、完全cross-fit必須 | **B** |
| **AP-oriented rank learner / MetaAP** | MetaAP 2022。citeturn21search3 | logloss分類でなくAP/orderを直接重視 | AUC/APが高い状況で残る微妙な順位を変える | 中〜高 | 高 | 中 | 小標本でmetric overfit | **B** |
| **Hard-example curriculum / reweighting** | LEAP上位解法。citeturn20search3 | 誤分類行だけに高weightを与える | 少数のhard companyを重点学習 | 高 | 中 | 低 | 同一OOFでhardness判定するとリーク | **B** |
| **PTaRL prototype representation** | ICLR 2024。citeturn15search7 | supervised prototype空間 | class境界をprototype距離で表現 | 中〜高 | 高 | 中 | 低 | **B** |
| **GRANDE differentiable tree** | ICLR 2024。citeturn15search11 | boostingでなくgradient-trained differentiable hard trees | GBDTとは異なる最適化とinstance-wise weighting | 中 | 高 | 中 | 低 | **B** |
| **MotherNet** | ICLR 2025。citeturn15search16 | small tabular向けhypernetwork生成NN | n=742には理論上面白い | 中 | 中 | 中 | external pretrained model規約 | **C** |
| **T-JEPA / train+test self-supervised pretraining** | ICLR 2025。citeturn15search1 | unlabeled joint representation | test 800行も表現学習に使える可能性 | 中 | 高 | 高 | **現在の規約未確認** | **C / HOLD** |
| **Pseudo-label self-training** | Santander、Feedback、Happywhaleで成功/失敗双方。citeturn13search9turn8search15turn9search2 | test predictionを再学習に使用 | confident testを追加できる | 低〜中 | 低 | 中 | **規約＋confirmation bias** | **C / HOLD** |
| **TabDiff / TabSyn synthetic augmentation** | ICLR 2024–25。citeturn15search3turn15search5 | minority synthetic generation | positive179件を増やせる可能性 | 中 | 高 | 高 | synthetic overfit | **C** |
| **Direct differentiable F1 / exact Fβ optimization** | sigmoidF1、2025 direct metric optimization。citeturn21academia49turn21academia50 | BCE→thresholdではなくF1 surrogateを学習 | metric alignment | 中 | 中〜高 | 中 | **metric overfit大** | **C** |

### 優先順位の意味

**Sランク**は、単体スコアではなく「exp032と違う順位・違う誤り」を作れる可能性を重視しています。特に SetFit、ModernNCA、AutoCross-LR は、それぞれ **意味空間・距離空間・交互作用基底**という別々のものを作ります。

**Aランク**には有望だが計算量またはoverfit riskが高い手法を置きました。Caruana GES は例外で、実装は軽いものの新情報を生み出さないため A です。

**Bランク**には「理論上は効くが、既存実験から見て重複しやすい」ものを置いています。特に hierarchical TE は、通常TEが既に負けているため知名度だけで再挑戦すべきではありません。fileciteturn5file0

**Cランク**は締切までの期待値が低いです。Pseudo-labeling は成功事例がある一方、Feedback Prizeでは CV改善がPublicに再現しない例もあります。加えて今回の adversarial validation はtrain/test差をほぼ検出できておらず、transductive adaptationの必然性が弱いです。citeturn8search15turn9search2 fileciteturn4file0

## 最優先候補の具体実験設計

ここでは「巨大実験」ではなく、**仮説を最小コストで殺せる設計**にします。

共通して、比較対象は **exp032** とします。単体指標だけでは採用しません。Performance は F1/AP/AUC、Diversity は probability Pearson・Spearman rank・threshold disagreement・error overlap、Incremental Value は exp032へ追加した ΔF1/AP/AUC、Stability はseed/fold/repeated split、Risk はfeature-selection leakageとthreshold overfitで判定します。exp023/025/034が示したように「単体Expert改善＝採用」は禁止です。fileciteturn2file0

また、`error_analysis_discovery.md` では既に discovery 482件 / lockbox 260件が分離され、lockboxを特徴発見に使わない方針になっています。これは維持すべきです。モデルやハイパーパラメータを discovery 内 repeated CV で固定した後にだけ lockbox を見る方が、過去の `has_DX` のような full-label feature-selection leakage を繰り返さずに済みます。fileciteturn3file0 fileciteturn6file0

**提案する共通 ACCEPT gate** は、現在のチーム基準を少し厳しくした次です。

| Gate | 基準 |
|---|---|
| Performance | development repeated split で平均 ΔAP ≥ +0.005 |
| Stability | AP改善 ≥11/15、F1改善 ≥9/15を理想 |
| F1 guardrail | 平均 ΔF1 > 0。最終metricなので明確な負なら採用しない |
| Diversity | exp032とのSpearman <0.95、またはerror overlapが明確に下がる |
| Incremental | candidate単体ではなく **exp032 + candidate** が改善すること |
| Lockbox | developmentで全仕様固定後に一度だけ確認 |

これは統計的な普遍則ではなく、現在チームが既に使っている「ΔAP 0.005＋70%勝率」という経験則を、今回の停滞局面用に強化した実験ルールです。fileciteturn6file0

### 教師あり target-aware text representation

**仮説**

現在最大の未探索領域です。

OpenAI embedding は「一般意味として似た文章」を近くします。しかし購入判定で重要なのは、一般意味ではありません。

例えば、

「DXを積極推進するが既に内製教育体制が成熟している会社」

と

「DX推進意欲が高く、これから全社員研修を外部支援で拡大したい会社」

は一般意味空間では非常に近くても、教育商材購入確率は違う可能性があります。

SetFit は labeled pair を使って SentenceTransformer 自体をfine-tuneし、同じlabelを近く、異なるlabelを遠くする表現を学習します。公式資料でも contrastive fine-tuning → classification head という二段構成になっています。少数label向けに設計されている点も742件という今回に適しています。citeturn16search0turn16search8turn16academia48

さらに2026年の MulTaBench は text-tabular 系 benchmark で、generic frozen embeddings より**target-aware tuningしたembeddingが有利になる状況**を体系的に検証しています。これは今回の「OpenAI frozen embeddingから次へ何をすべきか」に最も直接的な研究根拠です。citeturn16academia49

Foursquare 2022の3位解法も、text/numericレコードに対して XLM-R + ArcFace でラベル依存のmetric spaceを作り、その後 classifier で再判定していました。citeturn20search5

| 項目 | 設計 |
|---|---|
| 最初に使う列 | **今後のDX展望だけ**。最初から3文書融合しない |
| 第2段階 | DX展望で成功したら企業概要・組織図へ拡張 |
| Encoder | 規約上使用可能な日本語または多言語 SentenceTransformer |
| Objective | SetFit contrastive pair loss。positive/negative label pairをbalanced sampling |
| Head | Logistic Regression。NN headを増やさずまず低自由度 |
| 最小実験 | DX展望の SetFit score 1列だけ作り、E3 TF-IDFとの比較 |
| 次の実験 | E3の置換、およびexp032へ9本目Expertとして追加 |
| 比較 | E3、exp029のE4 frozen embedding、最終exp032 |
| Diversity | E3/E4/exp032とのSpearman、FN rescue率を見る |

**CV設計で最も重要なのは、encoderを全742ラベルでfine-tuneしてからCVすることを絶対にしないこと**です。それをやると representation 自体が validation label を見ています。

各外側foldについて、

`inner train labels → pair generation → SentenceTransformer fine-tune → validation embedding/prediction`

まで完全に閉じる必要があります。

**ACCEPT** は、まず discovery repeated CV で TF-IDF E3 比 AP +0.01程度の単体改善または明確なerror diversificationを確認し、その後 exp032へ加えて ΔAP ≥0.005、AP 11/15以上、平均ΔF1>0。

**REJECT** は、単体APが高くても exp032とのrank correlationが0.95超で、ensemble ΔAPがほぼ0なら終了です。これは exp034 GAMと同じ「強いコピー」です。fileciteturn2file0

この実験は **TOP1** です。

### TabPFN を新しいtabular priorとして入れる

TabPFN v2 は small-to-medium tabular dataset を主対象とし、Nature論文ではおおむね1万行以下・数百特徴程度までの問題で強い結果が報告されています。742行、元列42程度、加工後でも約100列という今回の規模は、GBDTよりむしろ TabPFN を試しやすい側です。citeturn14search0turn14search11

重要なのは「有名なtabular modelだから」ではありません。

今回の E0 LightGBM と E0b LR は既に強いですが、この2つはそれぞれ tree partition とlinear boundaryです。TabPFN は大量のsynthetic tabular tasksで事前学習された prior を使って予測するため、**同じ列から別の一般化バイアスを得られる可能性**があります。citeturn14search0

| 項目 | 設計 |
|---|---|
| 最初の入力 | textを除く構造化特徴だけ |
| 入れないもの | TF-IDF高次元、OpenAI embedding全次元 |
| 第2段階 | E3/E4のOOF score 2列だけ追加し、軽量late fusion |
| 比較対象 | E0、E0b、exp032 |
| 目的 | 単体bestではなくE0/E0bと異なる順位を作れるか |
| diversity | E0/E0bとのrank corr、特にE0誤りのrescue率 |
| leakage | 通常CVは低risk。ただしpretrained model使用可否を現行Competition Ruleで確認 |

**最小実験**では textを一切入れません。これで TabPFN の純粋なtabular diversityを数時間級の巨大探索なしで判定できます。

**ACCEPT** は exp032へ1 expert追加して ΔAP ≥0.005、かつ F1 guardrail正。

**REJECT** は TabPFN単体が強くても E0/E0bと correlation >0.95、追加 ΔAP <0.002なら即停止です。

ここでは hyperparameter tuning をほぼしない方がよいです。目的は「TabPFNを最高性能にすること」ではなく、**exp032と違う残差を持つかの検査**だからです。

### AutoCross 型の自動交互作用探索 + sparse LR

これは今回のデータに対してかなり高確率で当たり得ます。

理由は既にチーム自身の実験で証明されています。

E7「不満 × 財務」の12交互作用を LR に与えたところ、exp019比 ΔAP +0.0136、14/15勝という大きな改善になりました。一方、E0bでは同じ94列でもlinear modelにしただけで追加改善しています。fileciteturn2file0

この2つを合わせると、

> **木が内部で作るinteractionとは別に、「良いinteractionを明示基底としてlinear modelへ渡す」こと自体が、このコンペでは有効**

というかなり強い証拠があります。

AutoCross は high-order feature crosses を search し、linear/deep modelへ渡す枠組みとしてKDDで提案され、実ビジネスデータで評価されています。citeturn22search0

ただし最初から full AutoCross を再現する必要はありません。

| 項目 | 最小構成 |
|---|---|
| categorical atoms | 業界、上場種別、BtoB/BtoC、survey ordinal |
| numeric atoms | 従業員、売上、営業利益率、software投資比、資産等をfold内quantile bin |
| 候補 | まず2-way crossのみ |
| search | inner CVのAP改善でbeam/greedy selection |
| model | one-hot / hashed sparse Logistic Regression |
| 特徴数 | 最初は上位10〜30 crossまで |
| 比較 | E7、E0b、exp032 |
| 拡張条件 | 2-wayが通った場合だけ3-way |

狙うべきcrossは人手で先に決めすぎない方がよいです。ただし直感的には

`industry × size_bin`  
`industry × survey dissatisfaction`  
`listing × software investment_bin`  
`DX intent bucket × financial capacity bucket`

のような構造が候補になります。

重要なのは **crossの採否そのものが教師ありfeature selection** であることです。全742件で「効くcross」を選び、その後CVしてはいけません。outer validationとは独立したinner trainで選択するか、discoveryでcross setを固定してlockboxへ一度だけ適用します。

**ACCEPT** は E7を超える必要はありません。exp032への追加 ΔAP ≥0.005かつrank correlationが既存Expertより低ければ十分です。

**REJECT** は候補crossがfoldごとに完全に入れ替わる、またはE7との相関が0.95近くに張り付く場合です。

### ModernNCA による教師あり近傍空間

exp035 の embedding cosine-kNN が REJECT だったため、「kNNはもう試した」と考えるのは間違いです。

exp035 が試したのは、

> 汎用 OpenAI embedding 空間を固定  
> → cosine distance  
> → 近傍label平均

です。

ModernNCA はその逆で、

> **target labelを使って「何を近いとみなすべきか」自体を学習**

します。

ICLR 2025の ModernNCA は deep representation と Neighborhood Components Analysis を組み合わせ、大規模なtabular benchmarkで既存tabular deep learnerを上回り、CatBoostに匹敵する結果を報告しています。citeturn15search6

したがって、exp035とは本質的に別物です。

| 項目 | 設計 |
|---|---|
| 最初の入力 | structured featuresのみ |
| numeric | standardize＋missing indicators |
| categorical | one-hotまたは低次元embedding |
| text | **最初は入れない** |
| objective | supervised neighborhood / NCA loss |
| prediction | 学習後latent空間でtraining-fold neighborからpositive probability |
| 比較 | exp035固定kNN、E0、E0b、exp032 |
| class imbalance | class-balanced batch / samplingを使用 |
| leakage | validation row自身やvalidation labelsをneighbor databaseへ入れない |

特に見るべきなのは、exp035との違いです。

exp035 は既存モデルとの最大相関0.824と diversity は十分だったのに、単体AP 0.614で弱すぎて ensemble へ価値を出せませんでした。fileciteturn2file0

ModernNCA の成功条件は、

> **その0.82程度のdiversityを維持したまま、単体APを実用レベルまで押し上げられるか**

です。

これは非常に良い実験仮説です。

**ACCEPT** は exp032追加時 ΔAP ≥0.005。単体でE0を超える必要はありません。

**REJECT** は ModernNCA latent distance が結局 E0b / E7 の順位に収束する、または seed間の近傍が大きく崩れてincremental valueが再現しない場合です。

### Cross-fitted residual / correction model

「bestが間違えた企業だけ学習する」という考え方そのものはコンペで実績がありますが、今回いきなり hard-error classifier を作るのは危険です。

Child Mind 2位は stage-1 prediction を stage-2 GBDT が再スコアして改善しました。LEAP上位解法にも best model の誤りをhard sampleとして重点学習する発想があります。citeturn20search2turn20search3

一方、2026年の別Kaggle解法報告では、second-stage residual model が shuffled controlより悪く、error routing AUCも約0.60に留まり、confidence-gated fallbackも失敗した例があります。つまり、**「誤りがある」ことと「誤りが予測可能」なことは別です。** citeturn20search6

今回も exp028 でthreshold付近に明確な誤りclusterがなく、discoveryのFNには「非常に慎重なDX文章なのにpositive」と「全社員教育を明示するpositive」が混在しています。単一の簡単なerror ruleで直せる状況には見えません。fileciteturn2file0 fileciteturn3file0

したがって、最初の実験は correction model ではありません。

**最初に「残差が予測可能か」を検査します。**

各rowについて完全cross-fitted exp032 prediction `p_base` を作り、

\[
r_i = y_i - p_{base,i}
\]

または

\[
e_i = 1[\hat y_{base,i} \neq y_i]
\]

をtargetにします。

ただし correction features に既存94列を全部渡すのは避けます。そうすると単に exp032 を再学習するだけです。

使う候補は、

`target-aware text score`  
`ModernNCA local positive density`  
`Expert間 prediction std / disagreement`  
`E3とE4の差`  
`E7とE0の差`

など、**「base内部の不確実性または新しい表現」だけ**に絞ります。

その上で、

\[
\text{logit}(p_{\text{new}})
=
\text{logit}(p_{\text{base}})
+
g(z)
\]

という低自由度 correction をfitします。`g` は最初は ridge / small logistic correction 程度で十分です。

| 判定段階 | 条件 |
|---|---|
| Residual gate | error classifier AUCが repeated split で安定して >0.60〜0.62程度 |
| Signed residual | `g(z)` が residual と安定した関連を持つ |
| Full test | correction後にexp032よりAP/F1改善 |
| Reject | error predictabilityがほぼ0.5ならsecond-stage自体を即終了 |

最大の leakage point は `p_base` です。

correction model の学習行に使う `p_base` は、その行のlabelを見ていない**OOF predictionでなければなりません**。Bを評価する場合は、Aだけで base OOF→correction fitまで行い、A全体学習base→B predictionに correction を適用します。これは現在の `holdout_check.py` と同じ思想で実装できます。fileciteturn11file0

## 実験ロードマップ

時間制約下では、「最も面白いモデル」ではなく **期待値 ÷ 仮説検証コスト** の順に進めるべきです。

| 実験 | 内容 | 期待値 | 実装負荷 | 最初に何を判定するか | 次へ進む条件 |
|---|---|---:|---:|---|---|
| **Experiment A** | 既存clean OOF予測で diversity matrix＋Caruana GES | 高 | **低** | 現8 Expertのweight fitに未回収の組合せがあるか | exp032比AP/F1改善がrepeated splitで再現 |
| **Experiment B** | DX展望だけ SetFit / supervised contrastive | **最高** | 中 | target-aware text geometryに信号があるか | E3との差＋exp032 residual rescue |
| **Experiment C** | structured-only TabPFN | 高 | **低〜中** | foundation priorがE0/E0bと違う順位を作るか | corrとincremental APが通る |
| **Experiment D** | 2-way AutoCross-inspired + sparse LR | **高** | 中 | E7以外にもstable interaction basisがあるか | selected crossがfold安定＋ensemble改善 |
| **Experiment E** | structured-only ModernNCA | 高 | 中 | fixed kNN失敗をsupervised metricで克服できるか | exp035より単体性能を大幅改善しdiversity維持 |
| **Experiment F** | residual predictability diagnostic | 中〜高 | **低** | exp032の誤りに学習可能な構造があるか | error AUC / residual associationが安定 |
| **Experiment G** | gated logit correction | 高だが条件付き | 中 | residual modelのincremental value | Experiment F 通過時のみ |
| **Experiment H** | target-aware text＋tabular cross-attention TTT | 高 | **高** | intermediate fusionに固有信号があるか | SetFitがまず有効だった場合 |
| **Experiment I** | TabM | 中 | 中 | 新tabular NNのdiversity | TabPFN/ModernNCAが不発なら |
| **Experiment J** | calibration → nested F1 decision | 小〜中 | 低 | threshold分散がまだ残るか | calibration errorが明確に改善 |
| **Experiment K** | hierarchical TE / optimal binning | 中 | 中 | sparse linear Expertの補助基底 | AutoCrossが有効な場合のみ |
| **Experiment L** | pseudo-label / T-JEPA | 低 | 中〜高 | transductive gain | **ルール確認後のみ** |

### 最初にやるべき GES の位置づけ

Caruana Ensemble Selection はモデルライブラリから「今のensembleを最も改善するモデル」をgreedyに追加する方式で、単純な全モデルweight fitとは異なります。AutoMLでも広く用いられてきた post-hoc ensemble 法です。citeturn18search0turn18academia45

現在は nonnegative logistic meta が全Expertのlogitを同時にfitしていますが、履歴を見ると

- E1の新財務比は単体改善、合成悪化
- survey stepも単体改善、合成悪化
- GAMも単体改善、合成悪化
- E10はalpha変更の副作用

というケースが多いです。fileciteturn2file0

したがって一度、

`exp029 / exp030 / exp031 / exp032 / GAM / kNN / TabPFN / SetFit / ModernNCA ...`

の**同一split上の clean OOF predictions**をライブラリとして、forward selectionを行う価値があります。

ただし直接 F1 を greedy objective にすると742件では不連続metricへ貼り付きやすいので、最初は **APを主目的、F1をguardrail** とした方が安全です。現在のmeta alpha選択もAPを使っているのは同じ理由です。fileciteturn10file0

### Ranking は今すぐ最優先ではない

LambdaRank / pairwise rankingの発想自体は間違っていません。MetaAPのように imbalanced classification でAPそのものを重視するランキング手法も提案されています。citeturn21search3

ただ、現在 exp032 は既に holdout AUC 約0.955、AP 約0.883まで来ています。fileciteturn2file0

つまり「positiveを上位へ持ってくる能力」全体は相当に高いです。ここからさらにglobal rankingを少し改善しても、最終 threshold 周辺の数社が動かなければ F1 は変わりません。

したがって pure LambdaRank より、

> **違う表現で順位を作る SetFit / ModernNCA → その順位を既存ensembleへ追加**

の方が期待値は高いです。

Ranking model単体は Bランクで十分です。

### F1直接最適化も後回し

F1最適decisionには理論があり、確率が十分calibratedなら F1* と閾値の関係も導けます。citeturn21academia48

しかし現在は threshold ruleを既に複数比較しており、明確な改善は出ていません。fileciteturn2file0

Direct F1 surrogateや differentiable F1は objective alignmentとして魅力的ですが、742件では「数行を動かすとF1が大きく変わる」ため、CV metricへ貼り付く危険性があります。最近の direct Fβ optimization研究は存在しますが、今回の deadline で first-line にするほどの実戦根拠はありません。citeturn21academia49turn21academia50

やるならまず、

`exp032 score → inner-fold Platt calibration → calibrated score → threshold`

だけを試すべきです。これなら低自由度です。

### Semi-supervised は意外と優先度が低い

SantanderやHappywhaleにはpseudo-label成功例がありますが、Feedback Prizeではpseudo-labelでCVが改善してもPublic側で悪化したモデルがあり、最終ensembleへの採用が限定された例もあります。citeturn13search9turn9search2turn8search15

今回についてはさらに、

1. train/test adversarial AUC ≈0.5で目立ったshiftがない。fileciteturn4file0
2. exp032自体が既に強いため、pseudo-labelはほぼexp032自身の判断を複製する可能性が高い。
3. test 800件は unlabeled representation datasetとしてそれほど大きくない。
4. 現行2026大会のtest利用規約を公開検索から確認できていない。citeturn23search1turn23search2

という4点があります。

したがって **「testを使えるからpseudo-label」には行かない**方がいいです。S/Aランクを消化してからで十分です。

## 最終診断

### 現在のチームは何を過剰に探索しているか

率直に言うと、**「既に回収済みの情報を、少し違う特徴量・少し違うモデルで再抽出すること」を過剰に探索しています。**

具体的には、

**財務情報の再加工**は既にかなり深いです。利益率、財務健全性、従業員当たり、キャッシュフロー、ソフト投資、surveyとのinteractionまで触っています。exp023では財務Expert単体が改善してもensembleは悪化しました。fileciteturn5file0 fileciteturn2file0

**アンケートの再加工**も同様です。単純モデル、step特徴、GP/RBF-SVM/ARD-GPまで試し、12列では違うモデルでもほぼ同じ順位へ収束することまで確認されています。fileciteturn2file0

**既存座標上のモデル族変更**も飽和しつつあります。GAMは単体でLRより強かったのにensembleは悪化し、multi-seed平均もOOF APを上げながらPublic F1を0.8000→0.77419へ落としました。fileciteturn2file0

**threshold周辺の微調整**も既に十分です。複数threshold ruleを検証し、boundary error clusterも否定されています。fileciteturn2file0

ここからさらに Optuna、財務比率、survey transformation、GBDT variant を増やしても、**研究量の割に新しい残差信号はほとんど増えない**可能性が高いです。

### 逆に、ほとんど探索できていないもの

最大の穴は三つあります。

第一は **target-aware representation learning** です。

現在は

`text → TF-IDF / generic OpenAI embedding → classifier`

であって、

`text + purchase label → purchase-specific embedding space`

ではありません。GitHub上で教師ありcontrastive SentenceTransformer / SetFit / ArcFace / SupCon型表現学習は確認できません。fileciteturn2file0 fileciteturn9file0

第二は **supervised geometry / retrieval** です。

exp035は汎用embeddingで固定cosine-kNNをしただけです。ラベルから「企業Aと企業Bを近くすべきか」を学習する ModernNCA / TabR 型は別原理です。citeturn15search6turn15search0

第三は **自動的なinteraction basis discovery** です。

E7の手作り12 interaction が今までで最大級の成功だったにもかかわらず、その先を AutoCross 型に一般化していません。これはかなり大きな取りこぼしです。fileciteturn2file0 citeturn22search0

### 「別の強いモデル」を探す発想を捨てるべき理由

exp034 が決定的です。

GAM単体は E0b より強かった。それでもensembleに入れると負けました。fileciteturn2file0

これは、

> **単体CV性能そのものは、今や探索目的として不十分**

ということです。

今後の候補は最初から

\[
\text{Value}
\approx
\text{Performance}
\times
\text{Residual Diversity}
\times
\text{Stability}
\]

で見るべきです。

単体AP +0.02でも exp032とのSpearman 0.98なら価値は低い。

逆に単体APが多少弱くても、exp032のFNだけを安定して救えるモデルなら価値があります。

実際、特徴量グループ別 Expert 化で最初に価値を出した E2 は「単体最強だから」ではなく、他Expertと非常に違う信号を持っていたことが重要でした。現在のメタ実装自身も、弱いが直交するExpertの価値を意識して設計されています。fileciteturn9file0 fileciteturn10file0

### 今のbestからF1をもう一段上げる最有力の「異なる予測原理」

一つだけ選ぶなら、

> **購入ラベルを使った教師ありテキスト表現学習によって、企業文章の“購入タスク専用の意味空間”を作ること**

です。

モデル名で言えば SetFit / supervised contrastive SentenceTransformer が最初の実装候補ですが、本質はモデル名ではありません。

本質は、

> **これまで固定されていた「文章間の距離」を、購入ラベルに合わせて学習対象へ変えること**

です。

現在の TF-IDF は、

「どの文字・単語が似ているか」

を表します。

現在の OpenAI embedding は、

「一般意味として何が似ているか」

を表します。

LLM手作業特徴は、

「人間が事前に決めた数本の意味軸上でどこにいるか」

を表します。

しかし本当に欲しいのは、

> **「教育商材を買う企業として、この二社は似ているか」**

という距離です。

それは179件のpositive labelを使わなければ作れません。

SetFit は少数labelからcontrastive pairを大量生成できるので、742社しかないという弱点をある程度緩和できます。citeturn16search0turn16search8turn16academia48

2026年の multimodal benchmark でも generic embedding のままtabular learnerへ渡すより、target-awareにembeddingを調整することが有効な条件が示されています。citeturn16academia49

そして競技実例でも、Foursquare 3位は label-supervised ArcFace により raw text/numeric record の距離空間そのものを学習し、その後の二段階classifierにつなげています。citeturn20search5

何より、これは現在の成功パターンと一致します。

exp023 の「同じ財務をもっと賢く加工する」は失敗しました。

exp025 の「同じsurveyをもっと賢く加工する」も失敗しました。

exp034 の「同じ94列をもっと強いGAMで学ぶ」も失敗しました。

対して、

**新しい文章表現を持ち込んだ exp029、明示interactionという新しいbasisを作った exp030、線形という異なるdecision geometryを作った exp032 は成功しました。** fileciteturn2file0

したがって、次に必要なのは「さらに強い分類器」ではありません。

**次に必要なのは、新しい座標系です。**

その中で最もまだ掘れておらず、かつ現在の text + tabular というデータ構造に直接対応し、F1改善に必要な新しいresidual rankingを作れる可能性が高いのが、

> **教師あり target-aware text representation**

です。

現状の停滞を突破する一手として、私はこれを最優先に置きます。