# deep-research ロードマップ検証レポート

期間: 2026-08-15 〜 2026-08-16
対象: `documents/deep-research-report.md` が提示した S/A ランク候補
基準モデル: **exp032**（Public F1 0.80000 / OOF F1 0.8061 / ホールドアウト15rep AP 0.8830・F1 0.7891）
一次記録: `exp/hypothesis_log.md`、各 `exp/_*.json`
評価母集団: 固定 discovery 482件。**lockbox 260件は本レポート時点で未開封**

---

## 1. 要約

11件の候補を検証し、**exp032 に追加・置換して採用できたものはゼロ**。本命は exp032 のまま
変わっていない。第2枠には H44（陽性数固定 rank blend）を保持している。

得られた最も重要な知見は個々の REJECT ではなく、**負け方が3件とも同型だった**ことである。

> **ラベルを使って表現そのものを学習した候補は、対応する「固定された表現」に全て負けた。**

これはレポートが最優先に置いた仮説（target-aware representation learning）に対する、
明確な反証データである。同時に、残っている手がどれかを一意に絞る。

---

## 2. 検証結果一覧

| ID | 候補 | 単体性能 | exp032 への増分 | 結論 |
|---|---|---|---|---|
| R1 | target-aware text（SetFit型 / DX展望のみ） | AP 0.4452 / AUC 0.7339 | ΔAP -0.0022 / ΔF1 -0.0031 | REJECT |
| R2 | structured-only TabPFN（47列） | — | — | 保留（ライセンス待ち） |
| R3 | 残差予測性の診断 | error AUC 0.8399 | — | PASS |
| R5 | cross-fitted residual logit correction | — | ΔAP -0.0155 / ΔF1 -0.0044 | REJECT |
| R4 | structured-only ModernNCA | AP 0.3942 / AUC 0.6789 | ΔAP -0.0115 / ΔF1 +0.0081 | REJECT |
| H42 | AutoCross型 2-way cross + sparse LR | AP 0.6228 | ΔAP -0.0183 / ΔF1 -0.0115 | REJECT |
| R6 | Nested Caruana GES（15予測） | — | ΔAP -0.0146 / ΔF1 -0.0199 | REJECT |
| H43 | exp032-anchored R4 blend | — | ΔAP +0.0046 / 転送F1 +0.0011 | REJECT（Public 0.776119） |
| H44 | 陽性数固定 rank blend | — | ΔAP +0.0149 / 同一K F1 +0.0079 | 第2枠として保持（Public 0.781955） |
| H45 | 境界限定の双方向swap（R4+R1） | — | ΔAP -0.0019 / 固定K F1 -0.0472 | REJECT |
| H46 | H44 の交換数に上限 | — | ΔAP +0.0105 / 固定K F1 0.0000 | REJECT |

いずれも Stage 0（リーク確認・fold内で閉じた学習）を満たした上での数値である。

---

## 3. 中心的な発見: 学習した表現は、固定された表現に負ける

3つの独立した実験が、同じ形で失敗した。

| 実験 | ラベルで**学習した**表現 | 対応する**固定の**表現 | 差 |
|---|---|---|---|
| R1 | SetFit fine-tuned encoder → **AP 0.4452** | E3 TF-IDF → AP 0.5061 | **-0.061** |
| R4 | ModernNCA の学習済み距離 → **AP 0.3942** | exp035 固定 cosine-kNN → AP 0.5847 | **-0.191** |
| H42 | 探索で選んだ2-way cross → **fold間 Jaccard 0.098** | E7 の手作り12交互作用 → ΔAP +0.0136 (14/15) | 選択が再現しない |

各候補の設計は、レポートが指摘したとおり互いに別物である。R1 は意味空間、R4 は距離空間、
H42 は交互作用基底を作る。**別々のものを作ったのに、全て「学習した側が負ける」という同じ
結果になった。** これは個々のハイパーパラメータの問題として説明しにくい。

最も直接的な証拠は H42 の **fold間 Jaccard 0.098** である。325個の候補交互作用から
inner CV で選ばれる集合が、fold を変えるとほぼ完全に入れ替わる。選択という行為自体が
信号ではなくノイズを拾っている、ということを直接示している。

### 解釈

正例179件という予算を、**「表現空間を作る」のと「分類器を学習する」の二重に使っている**と
読むのが素直である。742行・正例179件では、後者だけでほぼ使い切っている。

R1 の設計は「fold内で encoder fine-tune まで閉じる」を厳守しており、これはリークを避ける
ためには必須だが、同時に **各 fold の encoder が見られる正例は約143件** ということでもある。
pair を大量生成しても独立企業数は増えない（この点はロードマップ作成時に明記していた）。

### 過去の成功例との対照

これまで採用できたものを並べると、**すべて「こちらのラベル予算をゼロで持ち込んだ表現」**である。

| 採用実験 | 表現 | 消費したラベル予算 |
|---|---|---|
| exp026 / exp029 | OpenAI `text-embedding-3-large` | ゼロ（外部で事前学習済み） |
| exp030 (E7) | 「不満 × 財務」という人間が決めた軸 | ゼロ（探索していない） |
| exp032 (E0b) | 同じ94列に one-hot + 線形境界 | ゼロ（新しい自由度なし） |

E7 については事後分解で、効いていたのは積12列の純増ではなく **Need / Capacity という親軸を
独立 Expert にしたこと**だと判明している。つまり効いたのは人間の事前知識であって、
交互作用という形式ではない。H42 が同じ形式を探索で再現しようとして失敗したことと整合する。

### 結論として言えること

deep-research レポートの「次に必要なのは、さらに強い分類器ではなく新しい座標系である」は
支持される。ただし、**その座標系を742行から学習しようとすると、学習コストが情報利得を上回る**
という制約が抜けていた。必要なのは「違う表現」ではなく **「ラベルを消費しない違う表現」**である。

---

## 4. 副次的な発見

### 4-1. 誤りが予測可能でも、補正できるとは限らない（R3 → R5）

exp032 の誤りを予測する分類器は **error AUC 0.8399**（shuffle対照 0.4498 ± 0.0936、
confidence-only 比 +0.0929）。誤りの位置には明確な構造がある。診断ゲートは正当に PASS した。

それでも低自由度 logit correction は **ΔAP -0.0155 / ΔF1 -0.0044**。FN救出4件に対して
新FN3件・新FP4件を作り、完全に相殺した。符号付き残差の予測力は最初から無く
（residual R² -0.082、Spearman -0.141）、「どの行を間違えているか」は分かっても
「どちら向きに直すか」は分からない、という状態だった。

診断ゲートの PASS を採用根拠にしてはいけない。

### 4-2. 単体性能ではなく残差の独自性を使う経路（R4 → H44）

R4 ModernNCA は単体 AP 0.3942 と最弱クラスだが、**exp032 との Spearman 0.302** は
これまでの全候補で最も低い。exp035（固定kNN）が最大相関 0.824 で「diversity は十分だが
単体が弱すぎた」ケースだったのに対し、R4 はさらに直交している。

そこで9本目の Expert として足すのをやめ、percentile rank だけを λ=0.12 で混ぜ、
**陽性数を exp032 と同じ K に固定**した（H44）。同じ素材で結果は次のように変わった。

```
                       AUC      AP       同一K F1
exp032                0.9401   0.8256     0.7559
H44 固定K rank blend  0.9381   0.8405     0.7638
差                   -0.0019  +0.0149    +0.0079
```

5 fold すべてが非ゼロ λ を選び、悪化 fold はゼロ、誤りは 62→60。事前登録した条件を
すべて満たした唯一の候補である。

派生の H45（境界限定 swap、固定K F1 -0.0472）と H46（交換数に上限、固定K F1 0.0000）は
どちらも条件を満たさず REJECT。H44 の形が効いていて、その周辺をいじると壊れる。

### 4-3. 陽性数の固定は、test 全体でしか保証されない（H43 / H44 の Public 実測）

H43 は blend 後に閾値が下がり、予測陽性が 236→249 に増加。test の変更13件が
**すべて `0→1` の片方向**になった。Public 0.776119 を復元すると TP=52 / FP=23 / FN=7 で、
Public 側に入った4件は**全部 FP**。exp036（5seed確率平均、Public 0.77419）も同型で、
あちらは削除だけの片方向だった。

H44 はこれを受けて陽性数を固定した。しかし Public 実測 0.781955 を復元すると:

```
                  TP   FP   FN   公開予測正例   Public F1
exp032            52   19    7        71        0.80000
H44               52   22    7        74        0.781955
差                 0   +3    0        +3        -0.018045
```

**test 全体では 10/10 の対称交換なのに、Public 240行の中では `0→1` が3件多く落ちた。**
TP は1件も動かず、FP だけ +3。分割が非公開である以上、これはどの固定K手法にも共通する
性質であり、H44 固有の欠陥ではない。

> 固定Kは**判定を公平にする手続き**であって、**本番での安全保証ではない**。

なお Public 差 -0.0181 は実測 std 0.043 の内側であり、これ単体では REJECT 根拠にならない。

---

## 5. 提出枠の現状

| 枠 | 候補 | 根拠 |
|---|---|---|
| 枠1 | **exp032**（`submission_exp032_seed42_20260815.csv`） | Public 0.80000、ホールドアウト15rep で全構成中最高。動かさない |
| 枠2 | **H44**（`submission_exp032_rankblend_r4_lam012_top236_seed20260815.csv`） | Discovery nested の事前規則を通過した唯一の候補。Public 0.781955 は std の内側で、Private 自動採用のため保持コストはゼロ |

差分は20行（`0→1` 10件 / `1→0` 10件）。Private 側には `1→0` が3件多く残る形になる。
test の実正例率は公開から逆算して約24.6%（約197行）で、本命の236件は正例側へ過剰なので
方向としては悪くないが、Public の交換が全外れである以上、積極的な追い風とは読まない。

**Public 結果を見て λ / K / 閾値 / 交換対象IDを再調整しない。** 240行・TP 1行単位の情報に
合わせにいくと確実に過学習する。

---

## 6. 残っている手

第3節の結論を適用すると、優先順位は一意に決まる。

| 順位 | 候補 | 状態 | 理由 |
|---:|---|---|---|
| 1 | **R2 structured-only TabPFN** | 実装完了・ライセンス待ち | 事前学習は数百万の合成タスクで完了しており、**こちらのラベル予算はゼロ**。今日の失敗パターンに構造的に当てはまらない唯一の候補 |
| 2 | H41 ローカルBERT埋め込み | 実装途中・判定未記録 | 固定エンコーダなので同様。ただし則5により、埋め込みが効くのは TF-IDF が弱い列だけで、組織図・企業概要は回収済み、DX展望は相関0.934 で不採用済み。伸びしろは薄い |
| 3 | 人間が決める新しい軸（E7型） | 未着手 | 探索ではなく事前知識で決める形なら生きている。ただし財務・アンケートの再加工は飽和と記録済み |
| — | SetFit の設定探索、ModernNCA の epoch/temperature 探索、AutoCross の3-way 拡張 | **再開しない** | 第3節の理由により、同じ形式のまま調整しても期待値が低い |

R2 の再開条件は以下3点で、いずれも性能の問題ではない。

1. 現行大会画面で外部 pretrained model の利用可否を確認する
2. ユーザー自身が `https://ux.priorlabs.ai` でライセンスを承諾する
3. API key を環境変数 `TABPFN_TOKEN` として設定する（ログ・Git に書かない）

---

## 7. このロードマップの成否について

ロードマップが定めた成功条件は「試したモデル数」ではなく次の4点だった。

- [x] target-aware representation が有効か、リークなしで結論を出す → **有効でないと結論**
- [x] S/A 候補3系統を低コストで判定する → R1 / R4 / H42 を discovery のみで判定、lockbox 未開封
- [x] exp032 を壊さず、通過した challenger だけを第2枠へ進める → H44 のみ
- [ ] 全候補が不発でも 8/30 までに探索を止め、再現性と説明資料を完成させる → 進行中

現状、ホールドアウト AP 0.8830 / AUC 0.9547 に対して、新しい表現を3系統試して残差が
取れなかった。**残差が単に取り切れない可能性**は真剣に考慮に値する。Public std 0.043 という
測定精度と、残り期間を踏まえると、TabPFN の判定後に探索を終了し、再現性と説明資料へ
資源を移す判断が妥当になる可能性が高い。

---

## 8. 生成物

| 実験 | スクリプト | 結果ファイル |
|---|---|---|
| R1 | `exp/target_aware_text.py`, `exp/compare_target_aware_text.py` | `exp/_r1_target_text_discovery_seed20260815.json` |
| R2 | `exp/compare_tabpfn.py` | （未実行） |
| R3 | `exp/diagnose_residual_predictability.py` | `exp/_r3_residual_predictability.json` |
| R4 | `exp/modern_nca.py`, `exp/compare_modern_nca.py` | `exp/_r4_modern_nca_discovery_seed20260815.json` |
| R5 | `exp/correct_residual.py` | `exp/_r5_residual_correction.json` |
| R6 | `exp/nested_ges.py`, `exp/compare_nested_ges.py` | `exp/_r6_nested_ges_discovery_seed20260815.json` |
| H42 | `exp/auto_cross.py`, `exp/compare_auto_cross.py` | `exp/_autocross_discovery_seed20260815.json` |
| H43 | `exp/compare_anchored_r4.py`, `exp/make_submission_anchored_r4.py` | `exp/_h43_anchored_r4_discovery_seed20260815.json` |
| H44 | `exp/compare_prevalence_rank_blend.py`, `exp/make_submission_prevalence_rank_blend.py` | `exp/_h44_prevalence_rank_blend_discovery_seed20260815.json` |
| H45 | `exp/compare_boundary_swap_h45.py` | `exp/_h45_boundary_swap_discovery_seed20260815.json` |
| H46 | `exp/compare_swap_cap_h46.py` | `exp/_h46_swap_cap_discovery_seed20260815.json` |

seed は全実験で 20260815。詳細な設定・fold別診断は各 JSON と `exp/hypothesis_log.md` にある。
