# data/ の中身（Git管理外）

SIGNATE配布データは再配布不可のため、このディレクトリは `.gitignore` している。
新しくクローンした人は、以下を共有ドライブから取得してここに置くこと。

## コンペ配布ファイル（SIGNATEからダウンロード）

| ファイル | 内容 |
|---|---|
| `train.csv` | 学習データ 742行（正例率 24.1%） |
| `test.csv` | 評価データ |
| `sample_submit.csv` | 提出フォーマット（ヘッダなし） |
| `description.csv` | 列の説明 |
| `train.xlsx` | 配布時のExcel原本 |

## 生成済みアーティファクト（共有ドライブから取得）

再生成にはAPIコストと時間がかかるので、作り直さず共有物を使うこと。

| ファイル | 内容 | 生成元 |
|---|---|---|
| `_emb_org_text-embedding-3-large.npz` | 組織図のOpenAI埋め込み（**exp026＝現本命が使用**） | `colab_openai_embed.py` |
| `_emb_overview_text-embedding-3-large.npz` | 企業概要の埋め込み | 同上 |
| `_emb_dx_outlook_text-embedding-3-large.npz` | 今後のDX展望の埋め込み | 同上 |
| `train_with_llm_3axes.csv` / `test_with_llm_3axes.csv` | H9 LLM3軸スコア（現在は未採用: exp021でREJECT） | 別途LLM実行 |

## 注意

- 埋め込みを作り直すと**値が変わる可能性がある**（モデル側の更新）。作り直したら
  `exp/repro_check.py` が落ちる。落ちた場合は環境が変わったということなので、
  過去のOOF記録と直接比較しないこと。
- `OPENAI_API_KEY` はコードに書かない。Colabのシークレット、または環境変数で渡す。
