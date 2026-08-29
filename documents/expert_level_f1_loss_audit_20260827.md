# Expert(E0)レベルF1近似loss監査 — 2026-08-27

## 結論

メタブレンド層で4連敗したF1近似loss（`documents/sigmoidf1_loss_audit_20260827.md`他）を、
1層下のExpert個別学習（E0_anchor、LightGBM）へ直接適用したらどうかを検証した。
strict holdout 5repで**3構成すべて、ほぼ全指標で明確な悪化**となり、メタ層より
はっきりしたREJECTだった。学習損失をF1へ寄せる方向は、この探索でどの層に置いても
効かないことが確認できた。

## 実装

`exp/lgbm_f1_objectives.py`にLightGBM用カスタム目的関数を追加した。

- `weighted_logloss_fobj(positive_weight)`: 正例を重み付けした通常logloss勾配・Hessian。
- `sigmoid_f1_fobj(beta, eta)`: メタ層と同じ`sigmoidF1`
  （Bénédict et al. 2021, arXiv:2108.10566, `S(u;beta,eta)=sigmoid(beta*(u+eta))`）を
  LightGBMのraw margin予測に直接適用。F1はサンプル全体の集計統計量で個体分離しないため、
  他個体のスコアを定数とみなす対角近似でHessianを導出し、非凸な損失面で負になりうる
  部分は`1e-3`で正の下限にクリップした（listwise/集計loss をGBDTへ載せる際の標準的手法）。

勾配は有限差分と最大`5.27e-10`で一致、Hessianも（クリップ前の値で）有限差分の
二階差分と厳密に一致することを確認した。

LightGBM 4.xでは`fobj`引数が廃止されており、`params["objective"]`にcallableを直接渡す
必要がある。カスタム目的はLightGBMの自動boost-from-averageを経由しないため、
`init_score`（学習foldの正例率のlogit）を明示的に持たせ、予測後に手動で加算し直す
（Max Halfordのfocal loss実装記事の作法に従った）。

`exp/compare_expert_f1_loss.py`で、E0_anchorの学習lossだけを差し替え、残り7 Expert・
outer split・inner fold・メタ損失（通常logloss）・alpha grid・cross-fitted Platt・
`half_f1`則はすべて現行本番構成と共有した。

## Screening 5rep

未使用seed 256〜260。

| E0学習loss | F1 | AUC | AP | LogLoss | MCC |
|---|---:|---:|---:|---:|---:|
| baseline logloss（現行） | **0.82125** | **0.96695** | **0.91596** | **0.20073** | **0.76331** |
| sigmoid_f1_b7e1 | 0.80718 | 0.96493 | 0.90727 | 0.21043 | 0.74616 |
| weighted_logloss_pos15 | 0.80856 | 0.96548 | 0.91294 | 0.20353 | 0.74743 |
| sigmoid_f1_b1e0 | 0.79766 | 0.95989 | 0.89817 | 0.22136 | 0.73418 |

### baseline logloss との差

| 候補 | ΔF1 | F1勝/同/敗 | ΔAP | ΔLogLoss |
|---|---:|---:|---:|---:|
| `e0_sigmoid_f1_b1e0` | -0.02359 | 1/0/4 | -0.01778 | +0.02064 |
| `e0_sigmoid_f1_b7e1` | -0.01408 | 1/0/4 | -0.00868 | +0.00971 |
| `e0_weighted_logloss_pos15` | -0.01269 | 1/0/4 | -0.00302 | +0.00281 |

3候補とも5repのうち4repでF1が悪化し、AUC/AP/LogLoss/Brier/MCCもほぼ全repで
劣化した。メタ層での結果（ΔF1はおおむね±0.006以内で拮抗）よりもはっきり悪い。

`weighted_logloss_pos15`は、過去の`scale_pos_weight=1.5`スモーク結果
（`documents/metric_and_loss_audit_20260827.md` 2.2節、fresh seed 201〜205で
ΔAP `-0.00229`・0/5改善、REJECT）と方向・大きさが一致しており、今回ゼロから
実装したcustom fobjが正しく動作していることの追加検証にもなった。

## 解釈

E0_anchorは1foldあたり実質380行前後・94列という小標本で学習するLightGBMである。
sigmoidFのHessian対角近似は非凸領域で頻繁にフロア値（`1e-3`）へ張り付き、Newton
ステップの情報量を落とす。集計統計量ベースのlossは個々のサンプルの勾配が他の
全サンプルの現在の予測値に依存するため、木を1本足すたびに全体最適点が動き、
GBDTの逐次的な葉最適化とは相性が悪い。低次元線形メタブレンド（8特徴量）では
この不安定性の影響が緩やかだったが、94列・数百本の木を持つLightGBMでは深刻に
効いたと考えられる。

## 採否

- **REJECT**: sigmoidF1・weighted logloss、いずれもExpert(E0)レベルへの適用。
- **KEEP**: 全ExpertでLightGBM binary loglossを維持。
- **実装として保持**: `lgbm_f1_objectives.py`・`compare_expert_f1_loss.py`は
  再利用可能な形で残す。

この監査から新しい提出CSVは作らない。これで「F1近似lossをどこかの層に挟む」という
方向は、メタブレンド層(4種)・Expert層(2種)の計6実装すべてでREJECTとなった。
