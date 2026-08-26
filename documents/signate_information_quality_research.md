# SIGNATE Student Cup：既存情報の品質向上に関する調査・仮説整理

掘り直した結果、前回の回答は少なくとも3点修正が必要です。

1. **DAEを有力候補として並べたのは甘かったです。** 今回とほぼ同じ標本サイズ・不均衡率のICRでは、Autoencoderによる次元削減は「promisingではなかった」というCompetition Masterの報告があります。
2. **「feature selectionが効く」も雑すぎました。** ICRでは2位は20特徴を削って成功した一方、3位や複数GMはfeature selectionそのものがtrainにoverfitしたため使っていません。つまり勝ち筋は「削る」ではなく、**削減方法の自由度まで下げること**です。
3. **今回のE0/E0bの差はかなり重要です。** これは単なる偶然ではなく、「木は候補splitの多さによるwinner's curseを受けやすいが、Ridge/L2-LRは複数の弱い重複信号を残すメリットがある」という構造で説明できます。

以下、前回よりかなり厳しく「本当に今回へ転用できるか」で切ります。

---

## 1. まず、今回に最も近いKaggleを見つけた

### ICR – Identifying Age-Related Conditions

これが今回のStudent Cupに一番近いです。

| 項目 | 今回のStudent Cup | ICR |
| --- | ---: | ---: |
| train | **742** | **617** |
| minority | **24.1%** | **17.5〜18%** |
| task | binary | binary |
| 特徴 | 約94+専門家 | **56特徴** |
| 特徴型 | heterogeneous | 主にnumeric |
| 問題 | CV分散・過学習 | **CV極端に不安定・過学習** |

Kaggle公式もICRを「minimal training」「number of samples is small」と明示しており、実際617行・正例17.5〜18%でした。まとめられた上位解法分析でも、「617 rows」「50+ features」「very unstable CV / easy to overfit」が最大の問題として挙げられています。[Kaggle](https://www.kaggle.com/competitions/icr-identify-age-related-conditions?utm_source=chatgpt.com)

そして上位3チームの行動が面白い。

### ICR 1位

**Feature Engineeringをほぼ捨てています。**

GBDTは明らかにoverfitし、feature engineeringもoverfitしたため、最終的にはTemporal Fusion Transformer全部ではなく、主として**Variable Selection Networkだけを3段積んだモデル**を使いました。Dropoutも0.75→0.5→0.25と非常に強い。10-foldの各foldを10〜30回再学習するほど学習分散が大きかったとされています。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

さらにCompetition Master Tiliiのコメントとして、

> autoencoderでdimension reductionを試したが、あまり有望ではなかった

という記録も残っています。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

これ、E1-DAEにかなり直接的な反証です。

### ICR 2位

逆にこちらは**特徴削減が効いています。**

56特徴前後からmanual permutationによって、

> その列を落としてscoreが少しでも良くなればdrop

という処理を行い、最良モデルでは**20特徴を削除**しています。UMAP→KMeansによる圧縮・cluster featureも試したものの、「あまりscoreをもたらさなかった」と報告されています。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

つまり、

**56 → 約36 features**

というかなり大胆なpruningは成功。

一方、

**nonlinear compression（UMAP）**

はほぼ効かなかった。

この差は今回かなり重要です。

### ICR 3位

さらに面白い。

ratio特徴も、targetとの相関Top-k feature selectionも試しましたが**最終submissionでは使っていません**。

まとめでは、

> NOT working on it too much and overfitting

が成功要因だったとされています。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

GibaもICRについて、

> feature selection overfitted trainset, so I didn't use it

と記録しています。[Kaggle](https://www.kaggle.com/titericz/writeups?utm_source=chatgpt.com)

したがってICRから得るべき教訓は、

> 「特徴削減は強い」

ではありません。

正確には、

> **小標本では「どの特徴を削るか」という探索自体がoverfitする。**  
> **だから、targetを使わない低自由度な削減、またはselectionをnested化する必要がある。**

です。

これは前回の回答で抜けていました。

---

## 2. Don't Overfit! IIはさらに極端

trainは**250行、300特徴**。

1位のZachの最終解法はかなり特殊でPublic LB probingを利用しているため、そのまま競技解法として移植するべきではありません。

ただし、非常に面白い実験があります。

300特徴それぞれに推定された弱い係数をそのまま利用するとPrivate 0.853程度。

そこで、

> `abs(coef) < 0.04 → 0`

と弱い信号を全部殺すだけで、

**Private 0.853 → 0.877**

まで上がりました。

さらにhard thresholdではなく、0近傍の係数を滑らかに0へ縮めるshrinkage関数を作ると、

**0.886**

まで上昇しています。[Kaggle](https://www.kaggle.com/competitions/dont-overfit-ii/writeups/zach-first-place-solution?utm_source=chatgpt.com)

一方4位はRFECVについて、

> `random_state`やnoiseを少し変えると選択特徴が変わりLBも落ちる

と報告しています。つまり、「特徴選択が見つけたbest subset」自体がsampling noiseだったわけです。[Kaggle](https://www.kaggle.com/competitions/dont-overfit-ii/writeups/feature-blind-4th-place-solution-no-lb-probing?utm_source=chatgpt.com)

今回への翻訳は明確です。

**「弱いsignalを発見する」より、「弱いsignalをゼロ方向に縮める」方が安全なフェーズに入っている可能性が高い。**

これはE6にかなり効く話です。

---

## 3. Porto Seguroを改めて読むと、DAEより重要なのは別のところ

### 1位 Michael Jahrer

これは有名なDAE解法ですが、細部が重要です。

最終ensembleは、

- 5 NN → **DAE hidden activations**
- 1 LightGBM → **raw data**

です。[Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/michael-jahrer-1st-place-with-representation-learn?utm_source=chatgpt.com)

DAEはswap noiseを使い、入力の約15%を他row由来の値へ置換して元データを復元させています。

しかし、

> **LightGBMにはDAEを使っていない。**

ここを飛ばして「DAEがtabularで効いた」と言うと間違います。

しかもPortoではunsupervised representation learningに**train+test**も使っています。巨大データでrepresentationを学習できる状況です。[Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/michael-jahrer-1st-place-with-representation-learn?utm_source=chatgpt.com)

今回：

- 742行
- Bは一切training不可
- E1は10〜15次元

です。

条件が違いすぎる。

### 3位 Utility

こっちの方が今回には重要です。

全`calc`特徴と複数の個別特徴を、

**1列ずつ削除 → LGB CVを見る**

というgreedy eliminationで除外しています。

そして最終的にかなりの列を落としています。[Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/utility-3rd-place-solution?utm_source=chatgpt.com)

さらにcategoricalはOHEした方が「useful categoryのsplit時のnoiseを減らせた」と書いています。[Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/utility-3rd-place-solution?utm_source=chatgpt.com)

つまりPortoの上位3解法だけでも、

**1位：NNはrepresentation、LGBはraw**  
**3位：LGBでもfeature eliminationは効く**

となっている。

「万能な最適特徴表現」はありません。

---

## 4. 金融データでも同じ現象がある

Home Credit Default Risk 1位は、集約処理などによって**1600以上**まで増えた特徴を、Ridgeベースの単純なfeature selectionで**約240特徴**まで落としています。

約85%削減です。[Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/overview/description?utm_source=chatgpt.com)

これはサンプル数が大きいため今回と同条件ではありません。

ただし、

> 「GBDTだから大量の不要特徴を勝手に無視する」

は少なくとも上位解法の経験則として成立しない。

という証拠にはなります。

---

## 5. IEEE-CIS Fraudは「何を圧縮するか」の良い事例

これも今回のE1/E6にかなり近い。

IEEE-CISには大量の`Vxxx`列があり、互いに強く相関・似たmissing patternを持っていました。

1位解法ではそれらをgroupにして、

- group内PCA
- 相互に非相関な代表subset
- group全体のmean

という**3種類の情報圧縮**を試しています。

そして最終的に時間安定性なども考慮してV列を削減しています。Kaggle公式solution documentationにも「V columnsを3 tricksでreduce」と記録されています。[Kaggle](https://www.kaggle.com/solution-write-up-documentation?utm_source=chatgpt.com)

これはE1にかなり直接移せます。

大事なのは、

> **全15財務列を一発PCA**

ではありません。

IEEEと同様、

> **冗長性の構造が同じfeature familyごとに圧縮する**

です。

---

## 6. Categorical Encodingでも「加工すればするほど良い」は否定される

Categorical Feature Encoding Challenge II 1位はもっと露骨です。

最終特徴処理は、

- 全特徴 → category
- Label Encoding
- ordinalだけOrdinal Encoding

**ほぼそれだけ。**

それ以外の多様な加工を試したが、

> they all lead to overfitting

だったと明記しています。[Kaggle](https://www.kaggle.com/competitions/cat-in-the-dat-ii/writeups/zetyun-com-dt-1st-place-solution?utm_source=chatgpt.com)

Porto 9位も、

> target encoding was not adding much

としています。[Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/mario-filho-kazanova-faron-9th-place-solution?utm_source=chatgpt.com)

したがってE2のLikert 1〜5に対して、

「TEをやって情報量を高めよう」

というのは、私はかなり反対です。

5カテゴリしかなく、しかもordinal structureがすでに分かっている。

そこへ742行から推定したtarget meanを持ち込むのは、

**denoiseではなくtarget noiseの注入**

になりかねません。

---

## 7. LLM蒸留については、検索結果を誇張しない方がいい

ここも前回甘かったです。

かなり探しましたが、

> **「1000行未満のKaggle/SIGNATE表形式binaryで、GPTに既存text/surveyを数個のsemantic scalarへ蒸留させ、それが上位解法の主要改善になった」**

という綺麗な公開winner writeupは、現時点では見つけられていません。

つまり、

**E0の親4軸成功をKaggleの定番手法として正当化するのは無理があります。**

むしろ現在のところ、

> あなた自身の  
> ΔAP +0.0198 / ΔF1 +0.0359

の実験結果の方が強い証拠です。

関連する研究としてはFeatLLMがあります。LLMにtabular feature/ruleを生成させ、simple linear classifierへ渡すfew-shot tabular学習で、13データセットに対して既存few-shot手法を平均的に改善しています。ただしこれは「compressionだけ」の研究ではありません。[arXiv](https://arxiv.org/abs/2404.09491?utm_source=chatgpt.com)

したがって、

**LLMをE2/E6へ使うなら、rowごとに好き勝手な特徴抽出をさせるのではなく、質問文・flag定義を一度だけ意味分類する**

使い方が妥当です。

ここは大きな違いです。

---

## 8. ここまでからE0/E0b現象を説明する

今回の

> E0 LGBM：94 raw → 約69 + 親4軸で改善  
> E0b LR：94 rawを残した方が良い

は理論的にも自然です。

### LightGBM側

たとえば同じlatent signal \(Z\) を表す、

\[
x_1,\ldots,x_{10}
\]

があるとします。

742行しかないにもかかわらず、木には各featureについて多数のthreshold候補があります。

すると、

> 本当に強いsplit

ではなく、

> **偶然Aで一番良く見えたsplit**

を拾う確率が上がります。

featureが重複しているほど、

**multiple comparisons / winner's curse**

が起こる。

したがって、

\[
x_1,\ldots,x_{10} \rightarrow z_{\mathrm{parent}}
\]

とすると、signalそのものを増やしていなくてもcandidate splitsを減らせる。

これがE0改善の一つの合理的説明です。

### Ridge / Logistic Regression側

一方、

\[
P(y=1)=\sigma(\beta_1x_1+\cdots+\beta_{10}x_{10})
\]

なので、L2 regularizationは各\(\beta_j\)を縮められます。

同じlatent constructを測る複数の**noisy measurement**があるなら、

\[
x_j = Z+\epsilon_j
\]

を複数残すことで、LR側がそれらを平均するような方向を学習できます。

つまりraw特徴は、

> redundant

であると同時に、

> **multiple noisy measurements of the same latent variable**

でもあります。

木では前者の害が大きい。

Ridge-LRでは後者のメリットが残る。

だから今回の逆転はかなり筋が通ります。

そしてこれが重要です。

- **E1 = LGBM**
- **E6 = LGBM**
- **E2 = LR**

なので、今回のE0経験則を転用するなら、

> **E1/E6は積極的に圧縮候補。**  
> **E2はかなり慎重。**

となります。

---

## 9. E1を本気で分解する

ここから事前登録形式にします。

### H-E1-1：財務特徴のtarget-free redundancy pruning

**出所**

- ICR 2位：56特徴から20特徴削除。
- Porto 3位：greedy feature elimination。
- IEEE-CIS 1位：相関feature familyのsubset化。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

**転用ロジック**

財務10〜15列は独立10〜15情報ではありません。

例えば概念上、

\[
Assets \approx Liabilities + Equity
\]

であり、

\[
ROA=\frac{Profit}{Assets}
\]

です。

売上・利益・ROA・利益率・資産等を全部渡すと、

**同じ企業規模・収益性を何度も違う座標系で渡している**

可能性があります。

これは今回E0の「manual flagsが同じ意味を重複表現していた」構造に近い。

**実装イメージ**

Aだけでtargetを一切見ず、

1. signed-logが必要な額を変換
2. Spearman correlation
3. `|ρ| >= 0.85`程度をedgeにする
4. connected componentをfeature familyとみなす
5. 各groupからrepresentativeを1列だけ残す

representativeはtarget APで選ばない。

例えば固定ルール：

**欠損率最小 → numerical stability → domain interpretability**

の順。

10〜15 → **6〜9列**程度。

E1 LGBMを完全置換して比較。

**想定リスク**

高相関でも、

\[
Profit/Assets
\]

のような「比率」がscale-free signalを持っている場合、raw amountだけ残すと消えます。

したがって**amount系とratio系を同じclusterにしない**方がいい。

**優先度：最高クラス**

ほぼノーパラメータで、E0の成功原理に一番忠実だからです。

---

## 10. H-E1-2：財務ブロックごとの潜在軸化

**出所**

IEEE-CIS 1位のgroup-wise PCA / mean / representative subset。[Kaggle](https://www.kaggle.com/solution-write-up-documentation?utm_source=chatgpt.com)

**転用ロジック**

global PCAではなく、

- Size
- Profitability
- Solvency
- Cash generation
- Investment intensity

のような**会計上同じ意味のblockだけ**まとめる。

例として、

\[
z_{\text{profit}}=\mathrm{PC1}(\text{OperatingMargin},ROA,ROE,\ldots)
\]

とする。

E0の

> 22 DX rules → 4 parent concepts

と数学的にはかなり似ています。

**実装イメージ**

Aだけで、

- robust median/IQR scale
- blockごとにPCA
- 原則PC1のみ

ただし無条件採用しない。

私は事前条件を入れます。

\[
ExplainedVariance(PC1)\ge 0.60
\]

かつbootstrapでloading方向が安定。

例えば30 bootstrapのsign alignment後、

\[
\operatorname{median}\,\cosine(loadings_b,loadings_{full})\ge 0.90
\]

を満たしたblockだけPC化。

満たさなければaggregateしない。

これで「PCAを試したらたまたま良かった」という自由度をかなり落とせます。

**リスク**

PCAは**yではなくvarianceを最大化**します。

minority特有のsignalが低variance方向ならPC1で消します。

したがってglobal PCAは反対。

**優先度：高**

ただしH-E1-1の後。

---

## 11. H-E1-3：財務block mean / medoid

これはPCAとは別にテストする価値があります。

**出所**

IEEE-CIS 1位が同じfeature familyを、

- PCA
- uncorrelated subset
- mean

の3方向で処理した点。[Kaggle](https://www.kaggle.com/solution-write-up-documentation?utm_source=chatgpt.com)

**転用ロジック**

742行なら、PCAのloadingそのものもsampling noiseです。

そこで、

\[
z_g=\frac{1}{|g|}\sum_{j\in g}z_j
\]

という**固定weight平均**を使う。

これはPCAよりはるかに自由度が低い。

E0の4軸がもし「複数観測からconcept scoreを取る」ことで効いたのであれば、こちらの方が再現しやすい可能性があります。

**実装**

block内robust-z → mean。

またはmedoid 1列。

比較は、

- `raw`
- `representative`
- `block mean`

の3つまで。

PCAまで一度に混ぜない。

**優先度：高**

小標本という意味ではPCAよりこちらを先にしてもいいです。

---

## 12. H-E1-4：E1 Denoising Autoencoder

**出所**

- Porto Seguro 1位：NNにDAE表現。
- ICR：Competition MasterがAE次元削減を試し「promisingではなかった」。[Kaggle](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/michael-jahrer-1st-place-with-representation-learn?utm_source=chatgpt.com)

**転用ロジック**

存在はする。

ただし今回は、

\[
n\approx500\sim600\quad（A側）
\]

に対して、

\[
p\approx10\sim15
\]

しかない。

この程度なら、

**非線形manifoldを学習する必要自体が疑わしい。**

**実装するなら**

極端に小さく、

\[
15\rightarrow8\rightarrow3\rightarrow8\rightarrow15
\]

swap noise 10〜15%。

Aだけでfit。

latent3のみ → E1 LGBM。

ただしこれはH-E1-1〜3を全部終えた後。

**リスク：非常に高い**

DAEのparameter estimation noise > 得られるdenoising benefit

になる可能性が高い。

**優先度：低**

前回より明確に格下げします。

---

## 13. E2は「11列だから圧縮」と考えると間違う

ここはかなり重要です。

E2はたった11列で、しかもLR。

つまり、

> dimensionality reductionが必要

という議論は弱い。

狙うべきは、

**dimension reductionではなくmeasurement error reduction**

です。

Likert質問とは、本来、

> latent attitudeを複数のnoisy questionで測る

構造だからです。

---

## 14. H-E2-1：アンケート semantic subscale averaging

**仮説名**

**E2 Likertの意味尺度蒸留**

**出所**

直接同条件のKaggle winnerによるLLM蒸留事例は確認できず。考え方としてはFeatLLMのschema-aware LLM feature generation、および今回のE0成功を主要根拠とする。[arXiv](https://arxiv.org/abs/2404.09491?utm_source=chatgpt.com)

**転用ロジック**

LLMに742企業を読ませるのではありません。

LLMへ渡すのは**11問の質問文だけ**。

例えば、

> 以下11問を、回答者のDX教育商材購買行動に関係し得る潜在概念として、意味重複のある設問同士に3〜4群に分類せよ。目的変数・回答値は一切使用しない。

と一度だけ分類。

例えば仮に、

- DX課題認識
- 組織準備度
- 人材不足
- 外部サービス受容性

となったとします。

mappingを**実験前にfreeze**。

その後、

\[
S_g=\frac{1}{|g|}\sum_{j\in g}\tilde{x}_j
\]

として3〜4個のsubscaleへ。

**E0との類似**

E0：

\[
22\ \mathrm{manual\ rules}\rightarrow4\ \mathrm{semantic\ parents}
\]

E2：

\[
11\ \mathrm{noisy\ items}\rightarrow3\sim4\ \mathrm{semantic\ parents}
\]

構造はかなり近いです。

**重要**

reverse questionがあれば方向を揃える。

**リスク**

LRはraw11をそのままlinear combinationできる。

つまり固定平均化すると、

\[
\beta_1=\beta_2=\beta_3
\]

に近い強い制約を入れることになります。

設問ごとのsignal strengthが違えば悪化します。

**優先度：中**

E1/E6ほど期待しません。

---

## 15. H-E2-2：Ordinal Factor Analysis

**出所**

相関群を潜在表現へ圧縮する上位解法としてIEEE-CIS等。今回はsurvey measurementというデータ構造上、PCAよりFactor Analysisの方が概念的に自然。[Kaggle](https://www.kaggle.com/solution-write-up-documentation?utm_source=chatgpt.com)

**転用ロジック**

モデルを、

\[
x_j=\lambda_jF+\epsilon_j
\]

と考える。

PCAは、

> varianceを説明する方向

ですがFAは、

> **共通潜在因子によって共有されるvariance**

を見る。

今回の「ノイズを消してcommon signalを取る」という目的にはFAの方が近い。

**実装**

Pearsonではなく、まずSpearman correlation。

可能ならpolychoric correlationを使う。

factor数をyで決めない。

- parallel analysis
- eigen spectrum
- loading stability

だけで決める。

例えば2〜3因子。

Factor scoreだけ → LR。

**リスク**

11問しかないとはいえ742回答なら推定自体は可能ですが、多因子を作るほどsampling varianceが増えます。

4因子以上にはしない方がいい。

**優先度：中**

semantic subscaleと独立に1回ずつ比較。

---

## 16. H-E2-3：Likert Target Encoding

これは提案ではなく、

**今はやらない方がいい仮説**

として登録したいです。

理由：

- levels ≈ 5
- ordinal meaningが既知
- E2=LR
- n=742
- positive≈179

例えばある回答levelに20人しかいなければ、

\[
\hat{p}(y=1\mid x=5)
\]

はかなり揺れます。

Nested OOF + Bayesian smoothingをしても、元の1〜5という低variance表現をtarget-dependentな値へ変えるメリットが小さい。

Cat-in-the-Dat II 1位も複雑なencodingを追加するとoverfitしたため、最終的にはLabel/Ordinal Encoding程度でした。[Kaggle](https://www.kaggle.com/competitions/cat-in-the-dat-ii/writeups/zetyun-com-dt-1st-place-solution?utm_source=chatgpt.com)

**優先度：低。**

---

## 17. E6が実は一番怪しい

依頼ではE1→E2→E6の順に挙がっていますが、

**実験期待値だけで順位を付けるなら、私はE6をE1以上に置きます。**

理由は単純で、

> 今回すでに削って成功した「DX手動辞書22列」

と**同じ人間設計の二値特徴群**だからです。

ここが最大の未処理ノイズ源である可能性があります。

---

## 18. H-E6-1：Rare / Duplicate flag elimination

**出所**

ICR 2位の20特徴drop、Porto 3位のgreedy elimination、IEEE-CISの冗長group処理。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

**転用ロジック**

binary flagはLGBMにとって危険です。

例えば742行中、

\[
flag=1: 8\text{件}
\]

しかなければ、

6 positive / 2 negative

という偶然だけで非常に強いsplitに見える。

しかも似たflagが10本あれば、

**最も偶然よく見えたものを木が拾える。**

典型的multiple testingです。

**実装**

targetを完全に見ず、各flagについて、

- support
- prevalence
- Jaccard
- phi correlation
- implication

を計算。

例えば、

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

`J > .8`

ならほぼ同義。

また、

\[
P(B=1\mid A=1)>0.95
\]

ならAがBのほぼsubset。

こうした関係だけからgroup化。

rare flagも、例えばA側support < 10〜15なら単独列としては削る／親groupへ吸収する。

閾値は**yを見て変えない**。

**リスク**

rareだが本当に強いsignalを消す。

ただし742行では、そのsignalと偶然を区別できないなら、むしろ消す方がgeneralization上合理的です。

**優先度：最高**

---

## 19. H-E6-2：Semantic family capped-count

これはかなり本命です。

**仮説名**

**E6手動flagの親概念蒸留**

**出所**

今回E0自身の成功 + IEEE-CIS group compression + Don't Overfit II shrinkage思想。[Kaggle](https://www.kaggle.com/solution-write-up-documentation?utm_source=chatgpt.com)

**転用ロジック**

例えば、

- IT人材不足flag
- DX人材不足flag
- 専門人材不足flag
- 教育不足flag

があったとします。

木には4つ渡さない。

\[
HumanResourceNeed=\min(flag_1+\cdots+flag_4,\ 2)
\]

のようにする。

なぜ`cap=2`か。

同義表現を4個踏んでいる企業が、

> 1個踏んでいる企業より4倍強い

とは限らないからです。

**synonym multiplicityをsignal strengthと誤認しないためのcompression**です。

候補は各familyについて、

- `any`
- `min(count,2)`

まで。

mean/count/weighted count/maxなどを全部探索するとまたoverfitします。

**リスク**

本当の累積効果が線形ならcapが情報を落とす。

**優先度：最高**

E0の再現性テストとして最も綺麗です。

---

## 20. H-E6-3：Nested stability selection

Permutation Importance / SHAP / Borutaを使うなら、私はこの形以外は勧めません。

**出所**

ICRでfeature selection成功例と失敗例が両方存在すること。Don't Overfit II 4位でRFECVの選択subsetがrandom seed依存だったこと。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

**実装**

外側：

\[
A_r/B_r,\quad r=1,\dots,15
\]

は現行のまま。

ただしA内部でさらに、例えば5×5 repeated stratified split。

flag \(j\) について、

\[
d_{jr}=Metric(Model_{all})-Metric(Model_{-j})
\]

をinner validationだけから取る。

15 outer Bはfeature selectionに絶対使わない。

そして例えば、

\[
P(d_j>0)\ge0.7
\]

のflagだけkeep。

つまり、

**「重要だったfeature」ではなく**  
**「重要性の符号がsamplingに対して安定しているfeature」**

を採用。

Boruta/SHAP magnitudeより、今の問題にはこちらの方が筋がいい。

**優先度：高**

ただし計算量は大きい。

---

## 21. ここで、15rep方式にも1つ盲点がある

これはかなり重要なので指摘します。

現在の

> A/B × 15回  
> Bは一切学習に使わない

は、**モデル学習のleakage防止としてはかなり強い**です。

しかしH30〜H51など数十個の仮説について、

**同じ15個のBの結果を見て次の仮説を考えている**

なら、その15 Bは研究プロセス全体では完全なholdoutではなくなっています。

形式的には、

\[
H_{t+1}=f(Result_B(H_1),\ldots,Result_B(H_t))
\]

となるからです。

つまりモデルにはBを入れていないけれど、

**研究者であるあなたがBへ適応している。**

これはadaptive overfittingです。

ICRの「overfitting yourself」という話そのものです。[Kaggle Solutions](https://kaggle.curtischong.me/competitions/ICR---Identifying-Age-Related-Conditions)

これまで5repで誤判定して15repへ増やした判断は正しい。

ただ、**仮説数が30、50と増えると15repでも万能ではありません。**

今回の圧縮フェーズでは、

### Development 15rep

普段の判断。

### Audit 5〜10rep

**今まで一度も結果を見ていないseed**を別途固定。

候補が1〜2個に絞れたときだけ開封。

の二段階にするとかなり強くなります。

締切まで時間がないなら、最低でも今回の「information-quality family」を終えた最後にだけfresh seedsでauditすることを勧めます。

---

## 22. なので「Permutation / SHAP / Borutaで重要特徴を消す」は本命ではない

前回はここを雑に扱いました。

今回のnでは、

\[
Importance_j
\]

そのものがrandom variableです。

特にSHAPは**学習済みモデルが覚えたものを説明しているだけ**なので、

> SHAPが低い → 真に不要

ではない。

Permutation Importanceも、

> validation sample ≈150

程度なら非常に高分散。

Borutaもforestのimportanceを何度も比較するため、小標本ではselection instabilityが残る。

なので優先順位は、

\[
\boxed{
Target\text{-}free\ redundancy
>
Nested\ stability\ selection
>
Single\text{-}CV\ SHAP/PI/Boruta
}
\]

です。

ここは今回かなり重要な結論です。

---

## 23. 3情報源を比較するとこうなる

| Source | 現モデル | ノイズの種類 | 圧縮期待 | 私の判断 |
| --- | --- | --- | ---: | --- |
| **E1 財務** | LGBM | 会計恒等式・scale・ratio重複 | 高 | **本命** |
| **E2 Likert 11問** | LR | measurement error・設問重複 | 中 | 慎重 |
| **E6 manual binary flags** | LGBM | rare split・semantic duplicate | **非常に高** | **最本命** |

ここから実験期待値を付けるなら、

\[
\boxed{
E6\ semantic\ compression
\gtrsim E1\ redundancy\ pruning
> E1\ domain\ aggregation
> E2\ semantic\ subscale
> E2\ FA/PCA
\gg E1\ DAE
}
\]

です。

これは前回とは少し順位を変えます。

---

## 24. 実際に次に回すなら、この順

| 順位 | prereg | 変更 | 目的 |
| ---: | --- | --- | --- |
| **1** | H52 | E6 rare/duplicate pruning | E0成功の最直接再現 |
| **2** | H53 | E6 semantic family capped-count | 20? flags → 数parent |
| **3** | H54 | E1 target-free redundancy prune | raw 10〜15 → 6〜9 |
| **4** | H55 | E1 block mean/medoid | latent signal averaging |
| **5** | H56 | E1 block PCA | nonlinearでなくlinear compression |
| **6** | H57 | E2 semantic subscales | 11 → 3〜4 |
| **7** | H58 | E2 ordinal FA | common-factor denoise |
| **8** | H59 | E6 nested stability selection | supervisedだがnested |
| **9** | H60 | E1 DAE | 最後の探索 |

特に**H52〜H58ではraw + compressedを同居させない**。

最初は必ず、

\[
Expert(raw)\quad vs\quad Expert(compressed)
\]

です。

そうしないと、

「compressionが効いた」

のか、

「derived featureを追加したから効いた」

のか分離できません。

---

## 25. 今回の調査で一番重要だった発見

単純に、

> 「上位勢もfeature selectionしていました」

ではありません。

むしろ逆です。

### ICR 1位

FEするとoverfit → **やめた**。

### ICR 2位

56列から20列 → **消して成功**。

### ICR 3位

ratioもtop-k selectionも → **最終的に捨てた**。

### ICRのGM

feature selection自体がtrain overfit → **使わなかった**。

### Don't Overfit II

弱い係数 → **0へ縮めた方が強い**。

### Porto 1位

NN → **DAE**、LGB → **raw**。

### Porto 3位

LGBでも → **greedy elimination**。

### Cat Encoding II 1位

複雑なencoding → **全部overfit**。

### IEEE-CIS 1位

相関V features → **PCA / representative / mean**。

この一見矛盾する結果を統一する考え方が、

### **Effective Degrees of Freedom Budget**

だと思います。

「特徴量はいくつか」ではない。

> **各expertに、何個の独立した判断機会を与えるべきなのか。**

E0の94列には、実質的に同じ意味なのに別々のsplit候補として存在する列が多すぎた。

それを約25列減らし4 parent axesへ蒸留したことで、

\[
\text{signal}
\]

を大きく落とさず、

\[
\text{model freedom}
\]

だけを落とした。

だから埋もれていた財務健全性やCFの弱いsignalまで復活した。

この解釈なら、その後の現象まで一本で説明できます。

---

そして、この解釈が正しければ次の改善は「優秀な新特徴量」を探すことではありません。

**E1/E6のeffective degrees of freedomをどこまで削ってもsignalが残るかを測ること**です。

特にE6は、E0で既に「手動で意味を細分化した特徴が木にとって有害だった」という実証が一度出ています。私はここを最初に叩きます。

なお、今回調べた中で最重要の一次資料はこれです。

- [ICR competition](https://www.kaggle.com/competitions/icr-identify-age-related-conditions?utm_source=chatgpt.com)
- [Porto Seguro 1st — Representation Learning](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/michael-jahrer-1st-place-with-representation-learn?utm_source=chatgpt.com)
- [Porto Seguro 3rd — Feature Elimination](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/utility-3rd-place-solution?utm_source=chatgpt.com)
- [Don't Overfit! II 1st — Shrinking weak signals](https://www.kaggle.com/competitions/dont-overfit-ii/writeups/zach-first-place-solution?utm_source=chatgpt.com)
- [Categorical Encoding II 1st](https://www.kaggle.com/competitions/cat-in-the-dat-ii/writeups/zetyun-com-dt-1st-place-solution?utm_source=chatgpt.com)

前回より調査すると、**「DAE/PCA/SHAPを試そう」という手法カタログ型の結論ではなく、「selection自体のoverfitを避けながらexpertの自由度を落とす」という方が今回の本質に近い**です。
