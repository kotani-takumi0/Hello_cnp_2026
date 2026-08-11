# チーム作業規約 — SIGNATE × TECH OCEAN Student Cup 2026

前提: **メンバーはColabで実験を回す / 最終判定と提出は1人（メンテナ）のローカル環境に集約する。**

このコンペのパイプラインは「確率の1e-6の揺れが提出ラベルに増幅される」構造を持つ
（`exp/repro_check.py` の冒頭を読むこと）。だから **数字を作った環境を明示できない実験は、
存在しないのと同じ**として扱う。以下はそのための最小限の約束事。

---

## 0. 3行まとめ

1. **Git = コードと記録**（1.1MB）。**Drive = データと生成物**（47MB、再配布不可）。**Colab = 計算資源だけ**（状態を持たせない）。
2. 受け渡しの単位は**ノートブックではなく関数1つ**。`hypotheses.REGISTRY` に登録し、判定は共通ハーネスが出す。
3. **Colabの数字はスクリーニング専用**。ACCEPT確定と提出ファイル生成はメンテナのローカルのみ。

---

## 1. セットアップ（全員、最初に1回）

```bash
git clone <repo>
cd SIGNATE_Cup_2026_DX
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 共有ドライブから data/ の中身を取得（data/README.md 参照）
python3 exp/repro_check.py          # ← 受け入れテスト。これが通るまで実験を始めない
```

`repro_check.py` の判定の意味:

| 判定 | 意味 | やってよいこと |
|---|---|---|
| **OK** | 参照とビット一致 | すべて。記録値と直接比較してよい |
| **WARN** | 提出ラベルは一致するが確率に微小なずれ | 粗い比較のみ。**ΔAP 0.0005級の微差判定は信用しない** |
| **NG** | 本命が再現しない | **結果を共有しない**。原因を潰す |

---

## 2. Colab での実験

### 2.1 ノートブックは「ドライバ」であって「実装」ではない

Colabノートブックは3セルで済ませる。ロジックは1行も書かない。

```python
# セル1: 環境
!git clone https://<TOKEN>@github.com/<org>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt

# セル2: データ（Driveから）
from google.colab import drive; drive.mount('/content/drive')
!ln -sfn "/content/drive/MyDrive/signate_cup_2026/data" data
!python exp/repro_check.py --fast        # 環境と決定性の確認

# セル3: 実験
!python exp/run_hypothesis.py H30
```

理由: ノートブックに実装を書くと、**それを .py に書き直す工程＝再実装**になり、
「良かったスコア」が再現する保証がなくなる。上の形なら書き直す対象が最初から存在しない。

### 2.2 メンバーの納品物 = 関数1つ + レジストリ1行

既存の契約（`exp/hypotheses.py`）をそのまま使う。

```python
def h30_xxx(train, X):
    """H30 [仮説の根拠を insights.md の記号で]: 何を主張する特徴か。"""
    X = X.copy()
    X["新特徴"] = ...          # 行単位の決定的変換のみ（リーク無し）
    return X

REGISTRY = {
    ...,
    "H30": ("一言説明", h30_xxx),
}
```

- **自前でCVループを書かない。** `run_hypothesis.py` が 15seed のペア比較を回し、
  `decision.py` の基準で ACCEPT / PARK / REJECT を自動で出す。
- fold依存の特徴（target encoding等）は `FOLD_REGISTRY` / `NEEDS_CV` 側に登録する。
- 乱数を使う推定器を足す場合は **`random_state` を必ず明示**する（未指定は提出が変わる）。

### 2.3 Colabの数字の扱い（最重要）

Colabはライブラリ構成もCPUもローカルと違うため、**絶対値は環境をまたいで比較できない**。

- **禁止**: Colabで出た OOF F1 と `experiments.md` の記録値を直接比べること。
- **必須**: ペア比較（baseline と 候補）は**同一セッション内で完結**させること。
  同じ環境内のΔなら環境差は相殺されるので意味を持つ。
- **役割分担**: Colab = スクリーニング（REJECT/PARKの足切り）。
  ローカル = ACCEPT確定・合成後のホールドアウト検証・提出ファイル生成。
- **提出ファイルはメンテナのローカルでのみ生成する。** Colabからは出さない。

---

## 3. Git 設計

### 3.1 何を追跡し、何を追跡しないか

| 対象 | 置き場所 | 理由 |
|---|---|---|
| `exp/*.py`, `documents/*.md` | **Git** | 差分に意味がある |
| `submission/*.csv` | **Git** | 提出の記録。**上書き禁止**（過去に事故あり）。Git管理下なら復元できる |
| `exp/_*.csv`（スコア表） | **Git** | 実験の記録。小さい |
| `data/`（配布データ・埋め込み） | **Drive** | 47MB。SIGNATE配布物は再配布不可 |
| `exp/_*.npz`（OOFキャッシュ） | 追跡しない | 再現可能な生成物 |
| `exp/repro_reference.npz` | **Git** | 受け入れテストの参照値。小さいので例外的に追跡 |

### 3.2 ブランチ

- `main` = 本命の状態。**直pushしない**。
- 1仮説 = 1ブランチ: `exp/h30-<slug>`。
- **実験IDは着手時に予約する**（`experiments.md` の表に「実行中」行を先に立てる）。
  並行作業で採番が衝突すると記録が壊れる。

### 3.3 PRに必ず貼るもの

1. `python exp/run_hypothesis.py H30` の判定出力（ACCEPT/PARK/REJECTの行まで）
2. `python exp/repro_check.py --fast` の判定（**どの環境で出した数字かの証明**）
3. 既存エキスパートとの相関（0.9超えなら「同じ情報の別表現」なので却下。exp021の前例）

メンテナは PR をローカルで**再実行して**判定を確定する。Colabの判定は参考値。

### 3.4 ノートブックをコミットする場合

探索の記録として残すのは構わないが、**出力を消してから**コミットする。

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/xxx.ipynb
```

出力込みのipynbは (a) diffが読めない (b) **配布データの中身が出力に載るとGit経由での再配布になる**
という2つの問題がある。`notebooks/` の中身は「証拠」であって「実装」ではない。

### 3.5 コンフリクトの主戦場: `documents/experiments.md`

サマリー表を複数人が同時に触ると必ず衝突する。回避策:

- 表への追記は**自分の実験IDの1行のみ**。他人の行を整形しない。
- 詳細は表ではなく `## 実験詳細` に追記する（末尾追記なら衝突しにくい）。
- 衝突したら**両方の行を残す**方向でマージする。実験記録は消さない。

---

## 4. 判定と提出のルール（全員共有）

このコンペで既に焼かれた経験則。破ると同じ穴に落ちる。

- **単seedのOOF差で判定しない。** 乱数1本足しただけで15seed中4回F1が改善して見える。
- **エキスパート単体のACCEPTで採用しない。** 合成後のホールドアウトで負けた例が3つある
  （exp020 / exp023 / exp025）。`holdout_check.py` のF1ガードレールまで見る。
- **AP改善だけで採用しない。** 評価指標はF1。exp020はAUC/AP増でF1が 0/5 で負けた。
- **Public差 0.03未満は判断に使わない。** Publicは240行、実測σ=0.043。
  差を見たら必ず「何行分か」に換算する（手順は `experiments.md` のメモ）。
- **提出枠は共有資源。** 誰がいつ何を出すかはメンテナが管理する。
  提出は検証手段ではなく、最終2枠のヘッジのために使う。

---

## 5. 変更してよい / いけない

- `harness.py` / `decision.py` / `repro_check.py` は**判定の土台**。変更はメンテナのみ。
  ここが人によって違うと、全員のΔが比較不能になる。
- `hypotheses.py` への**追記**は誰でもしてよい（既存関数の変更は不可）。
- `requirements.txt` のバージョンを上げるときは、必ず `repro_check.py` をフル実行してから。
  通らなくなったら「環境が変わった」ということなので、過去の記録と直接比較しない。
