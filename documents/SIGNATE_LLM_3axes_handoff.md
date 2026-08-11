# SIGNATE企業購入予測：LLM 3軸特徴量 引継ぎ資料

作成日：2026年8月4日  
検証対象：

- `train_with_llm_3axes.csv`
- `test_with_llm_3axes.csv`

## 1. この資料の目的

企業の文章情報からOpenAI APIで構造化したLLM特徴量について、
生成済みCSVの整合性、特徴量の定義、これまでの判断、次に行う実験を
別のLLMまたは担当者へ引き継ぐ。

**結論：アップロードされたtrain/testファイルは、現在確認できる範囲で正常に生成されている。**
LLM特徴量の再生成ではなく、既存のH8bモデルへ3軸を追加した
複数seed CVから作業を再開すること。

---

## 2. コンペ・データの概要

- タスク：企業が対象商品・サービスを購入するかの二値分類
- 目的変数：`購入フラグ`
- 評価指標：F1
- train：742行
- test：800行
- trainの陽性：179件
- trainの陰性：563件
- 陽性率：24.1240%

### これまでのモデル状況（ユーザー報告値）

- LightGBM表形式ベースライン  
  Public F1：0.6029、OOF F1：0.6667、閾値：0.220
- 利益率などの財務派生特徴  
  Public F1：0.6111
- TF-IDF stacking（`今後のDX展望`）  
  Public F1：0.6838、OOF F1：0.7500
- TF-IDF stacking（`今後のDX展望`＋`企業概要`、H8b）  
  **Public F1：0.7040（現時点のPublic最高）**、OOF F1：0.7236

上記スコアは今回のCSVから再計算したものではなく、過去実験のユーザー報告値。

---

## 3. LLMへ渡した入力

各企業について、LLMに渡した企業固有の入力は次の3列だけ。

1. `企業概要`
2. `組織図`
3. `今後のDX展望`

形式：

```text
以下の企業文章から、指定された状態特徴を評価してください。

【企業概要】
{企業概要}

【組織図】
{組織図}

【今後のDX展望】
{今後のDX展望}
```

### LLMへ渡していない情報

- `企業ID`
- `企業名`
- `購入フラグ`
- 財務数値
- アンケート1〜11
- TF-IDFスコアや特徴語
- 業界、上場種別

`企業ID`はAPI入力には使わず、生成結果を元行へ結合するキーとしてのみ使用。

### API条件

- Responses API
- モデル固定：`gpt-4.1-mini-2025-04-14`
- Structured Outputs（JSON Schema、`strict=True`）
- 同一プロンプト・同一スキーマをtrain/testに適用
- LLMには購入確率や購入フラグを直接予測させない

---

## 4. 元のLLM出力と3軸への圧縮

LLMは元々、以下の8項目を0〜4で出力した。

- `dx_investment_strength`
- `plan_specificity`
- `implementation_urgency`
- `companywide_scope`
- `external_vendor_need`
- `organization_readiness`
- `cautiousness`
- `confidence`

高相関だったため、モデル投入用には次の3軸へ圧縮した。

### 4.1 `llm_dx_execution_score`

以下5項目の単純平均：

```python
llm_dx_execution_score = mean(
    dx_investment_strength,
    plan_specificity,
    implementation_urgency,
    companywide_scope,
    organization_readiness,
)
```

意味：DXへの投資意欲、計画具体性、実施時期、全社展開、
組織準備度をまとめた「DX積極性・実行力」。

元値が0〜4の整数5個なので、出力は0.2刻みになる。

### 4.2 `llm_external_vendor_need`

元の`external_vendor_need`をそのまま利用。  
意味：外部サービス、ベンダー、パートナー、コンサルティング等を
必要とする度合い。0〜4の整数。

### 4.3 `llm_cautiousness`

元の`cautiousness`をそのまま利用。  
意味：投資抑制、限定実証、段階導入、先送り等の慎重さ。0〜4の整数。

### モデルに入れない項目

- 元の8項目
- `confidence`
- `evidence_summary`

これらは監査・品質確認用で、現在のmerged CSVには含まれていない。

---

## 5. アップロード済みファイルの検証結果

### 5.1 構造

| 項目 | train | test |
|---|---:|---:|
| 行数 | 742 | 800 |
| 列数 | 46 | 45 |
| `購入フラグ` | あり | なし |
| 企業ID欠損 | 0 | 0 |
| 企業ID重複 | 0 | 0 |
| 完全重複行 | 0 | 0 |
| 3軸欠損 | 0 | 0 |
| 3軸の無限値 | 0 | 0 |

- train企業ID：0〜741、連続
- test企業ID：742〜1541、連続
- train/testの企業ID重複：0件
- testの列はtrainから`購入フラグ`だけを除いた構造と完全一致
- 3軸は両ファイルの末尾3列として追加済み

### 5.2 特徴量の値域

| 特徴 | train最小 | train最大 | test最小 | test最大 |
|---|---:|---:|---:|---:|
| `llm_dx_execution_score` | 0.4 | 4.0 | 0.4 | 4.0 |
| `llm_external_vendor_need` | 0 | 4 | 0 | 4 |
| `llm_cautiousness` | 0 | 4 | 0 | 4 |

- `llm_dx_execution_score`は0.2刻みで19種類
- 他2軸は0〜4の5種類
- 不正な小数刻み、範囲外、欠損は確認されなかった

### 5.3 train/test分布差

標準化差の絶対値が0.2未満なら、一般に小さい差とみなせる。
3軸すべて0.14未満だった。

| 特徴 | train平均 | test平均 | test−train | 標準化差 |
|---|---:|---:|---:|---:|
| `llm_dx_execution_score` | 2.4458 | 2.4583 | +0.0124 | +0.013 |
| `llm_external_vendor_need` | 2.2736 | 2.3775 | +0.1039 | +0.131 |
| `llm_cautiousness` | 2.9946 | 2.9900 | -0.0046 | -0.004 |

**判断：3軸について大きなtrain/test分布シフトは確認されない。**

### 5.4 3軸間の相関

train：

- 実行スコア × 外部需要：0.686
- 実行スコア × 慎重さ：-0.764
- 外部需要 × 慎重さ：-0.510

test：

- 実行スコア × 外部需要：0.662
- 実行スコア × 慎重さ：-0.747
- 外部需要 × 慎重さ：-0.446

実行スコアと慎重さには強い負相関があるが、
完全な反転ではないため、両方を残してLightGBMで検証する。

---

## 6. 目的変数との予備的な関係

以下は全trainを使った記述統計であり、CV性能ではない。

| 特徴 | 未購入平均 | 購入平均 | 購入−未購入 | target相関 | 単変量AUC※ |
|---|---:|---:|---:|---:|---:|
| `llm_dx_execution_score` | 2.2171 | 3.1654 | +0.9483 | +0.4087 | 0.7797 |
| `llm_external_vendor_need` | 2.1652 | 2.6145 | +0.4493 | +0.2367 | 0.6460 |
| `llm_cautiousness` | 3.2274 | 2.2626 | -0.9648 | -0.3786 | 0.7323 |

※慎重さは低いほど購入側なので、AUCは向きを反転した値を記載。

観察：

- 購入企業は`llm_dx_execution_score`が高い
- 購入企業は`llm_external_vendor_need`が高い
- 購入企業は`llm_cautiousness`が低い
- 3軸はいずれも期待方向の関係を示した
- 特に実行スコアは単変量でも比較的強い分離を示す

ただし、LLM特徴の採否は必ず交差検証で判断すること。

---

## 7. 次に行う実験

### 最優先

現在のH8bに3軸だけを追加する。

```python
LLM_3AXES_COLUMNS = [
    "llm_dx_execution_score",
    "llm_external_vendor_need",
    "llm_cautiousness",
]
```

比較対象：

```text
M0: H8b
M1: H8b + LLM 3軸
```

### 評価方法

単一seed・単一閾値だけで判断しない。

最低限記録するもの：

- 複数seedのOOF F1平均・標準偏差
- OOF Average Precision平均・標準偏差
- 各seedでM1がM0を上回ったか
- 最適閾値の平均・標準偏差
- OOF予測の相関
- Public提出スコア
- 学習時間、特徴数

### 閾値

F1は閾値依存なので、各foldのvalidation内で閾値を探索した値を
外側評価へ漏らさない設計が望ましい。少なくとも、
同じOOF予測に対して都合よく閾値を繰り返し最適化し、
結果だけを選ぶことは避ける。

### アブレーション候補

M1が不安定な場合だけ以下を追加：

```text
M2: H8b + llm_dx_execution_score
M3: H8b + llm_external_vendor_need
M4: H8b + llm_cautiousness
M5: H8b + 実行スコア + 外部需要
M6: H8b + 実行スコア + 慎重さ
```

最初から大量の組合せを試さず、M0とM1を先に評価する。

---

## 8. 読み込み例

```python
import pandas as pd

train = pd.read_csv("train_with_llm_3axes.csv")
test = pd.read_csv("test_with_llm_3axes.csv")

target_col = "購入フラグ"

llm_features = [
    "llm_dx_execution_score",
    "llm_external_vendor_need",
    "llm_cautiousness",
]

assert train["企業ID"].is_unique
assert test["企業ID"].is_unique
assert train[llm_features].notna().all().all()
assert test[llm_features].notna().all().all()

X = train.drop(columns=[target_col])
y = train[target_col]
X_test = test.copy()
```

既存H8bノートブックが元のtrain/testを別途読み込む構成なら、
`企業ID`で3軸だけをmergeしてもよい。ただし、今回のCSVは既に
元データへ3軸が結合済みなので、そのまま読み込む方が簡単。

---

## 9. 注意事項

1. **LLM特徴生成時に購入フラグは渡していない。**  
   ラベルリークではない。

2. **固定zero-shot抽出なのでOOF生成は不要。**  
   ラベル由来情報を入力していないため、全train/testを同じプロンプトで
   一度処理した特徴を利用できる。

3. `llm_dx_execution_score`と`llm_cautiousness`は高相関だが、
   企業によっては「実行力も慎重さも高い」状態があるため、
   CV前に片方を削除しない。

4. `confidence`は文章量・具体性の代理になり得るため不採用。

5. 3軸の予備的なtarget差は良好だが、
   全trainで見た記述統計なので性能保証ではない。

6. 次のLLMは、LLM特徴を再生成・再設計するより先に、
   H8bへの追加効果を複数seed CVで測ること。

7. H8bにはTF-IDF由来特徴があるため、
   LLM特徴が文章情報を重複している可能性がある。
   単体性能ではなく、H8bへの**追加価値**を評価する。

---

## 10. 次のLLMへ渡す短い指示

```text
添付のtrain_with_llm_3axes.csvとtest_with_llm_3axes.csvを使って、
現在のベストモデルH8b（企業概要＋今後のDX展望のTF-IDF stacking）に
以下3列を追加したモデルを作成してください。

- llm_dx_execution_score
- llm_external_vendor_need
- llm_cautiousness

まずH8b単体とH8b＋3軸を同一CV分割・複数seedで比較してください。
評価はOOF F1、Average Precision、最適閾値の平均と標準偏差、
改善seed数を記録してください。

LLM特徴は購入フラグを入力せず生成済みで、再生成は不要です。
元の8特徴やconfidenceは使用しません。
単一seedのF1だけで採否を決めないでください。
```

---

## 11. ファイル識別情報

- train SHA-256：`273c6d8b503bc36baed1ecfbc9e150ba41b5e8938ef023e922bffddc32b7665e`
- test SHA-256：`d3aee682ee451e3c3026fdc38098708864ab8431bbea58aea38ca1162fc4b318`

機械可読な詳細検証結果は`llm_3axes_validation.json`を参照。
