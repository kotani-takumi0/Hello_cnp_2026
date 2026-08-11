"""OpenAI Embeddings API でテキスト3列をベクトル化して npz に落とす（Colab実行用）。

**このスクリプトは学習を一切しない**。ラベルを見ない行単位の決定的変換だけを行い、
結果をキャッシュする。以降の CV / ホールドアウト検証はすべてローカルで、
API を1回も叩かずに回せる。

なぜこの分離にするか:
  - embedding は教師なしの行変換なので fold 内 fit が不要（リークしない）。
    ネストCVの中に入れる必要があるのは、この上に載せる LR だけ。
  - API 呼び出しは決定的とは限らない（モデル改訂やサーバ側の差で 1e-6 は動きうる）。
    このプロジェクトは LR の random_state 未指定で提出が2行ズレた前科がある
    （documents/experiments.md「再現性の修正」）。**一度取ったら再取得しない**、
    npz を唯一の真実にする、という運用で揺れを構造的に消す。

次元について:
  text-embedding-3 系は Matryoshka 学習済みなので、3072次元で保存しておけば
  「先頭 k 次元を取って L2 再正規化」だけで 256 / 512 / 1024 次元版が作れる。
  **API を再課金せずに次元をハイパラにできる**ので、ここでは常に最大次元で保存し、
  切り詰めはローカル（exp/embedding_features.py）で行う。

使い方（Colab）:
  1. 左メニューの鍵アイコン（シークレット）に `OPENAI_API_KEY` を登録し、
     このノートブックからのアクセスを ON にする。コードに直書きしないこと。
  2. %cd /content/drive/MyDrive/path/to/SIGNATE_Cup_2026_DX
  3. !pip install -q openai
  4. !python3 colab_openai_embed.py --project . --yes

出力（既定）:
  exp/_emb_dx_outlook_text-embedding-3-large.npz
  exp/_emb_overview_text-embedding-3-large.npz
  exp/_emb_org_text-embedding-3-large.npz
  各 npz: train (n_train, d) / test (n_test, d) / train_ids / test_ids / meta
"""
import argparse
import os
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

ID_COL = "企業ID"
DEFAULT_MODEL = "text-embedding-3-large"

# slug は npz のファイル名に入る。列名を直接使うと日本語ファイル名になるので固定の別名を持つ。
COLUMNS = {
    "dx_outlook": "今後のDX展望",
    "overview": "企業概要",
    "org": "組織図",
}

# 概算単価（USD / 1M tokens）。表示のためだけに使う。
PRICE_PER_1M = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


def get_api_key():
    """Colab シークレット → 環境変数 の順に探す。コードへの直書きは受け付けない。"""
    try:
        from google.colab import userdata  # noqa: F401

        key = userdata.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY が見つからない。Colab のシークレットに登録するか、"
            "環境変数に設定すること（コードへの直書きは禁止）。"
        )
    return key


def normalize(text):
    """NFKC と空白の圧縮だけ。数字の 0 潰しはしない。

    TF-IDF 側（text_features._normalize）は数字を 0 に潰しているが、embedding は
    「従業員数百名規模」のような数量表現も意味として拾えるので潰さない。
    改行は意味の切れ目なので空白1つに落とすに留める。
    """
    s = unicodedata.normalize("NFKC", str(text))
    return " ".join(s.split()).strip()


def load_texts(project, col):
    """train / test の指定列を正規化して返す。企業ID も揃えて返す。"""
    out = []
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(project, "data", f"{split}.csv"))
        texts = df[col].fillna("").astype(str).map(normalize).tolist()
        assert all(t for t in texts), f"{split}/{col} に空文字がある"
        out.append((texts, df[ID_COL].values))
    return out


def embed_batch(client, model, texts, retries=6):
    """1バッチを埋め込む。レート制限・一時エラーは指数バックオフで再試行する。"""
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            # API は index 順を保証するが、念のため昇順に並べ直す
            rows = sorted(resp.data, key=lambda d: d.index)
            return np.asarray([r.embedding for r in rows], dtype=np.float32)
        except Exception as e:  # noqa: BLE001 - SDK の例外階層に依存したくない
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    再試行 {attempt + 1}/{retries - 1} ({type(e).__name__}) {wait}s待機",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError("到達しない")


def embed_all(client, model, texts, batch_size):
    parts = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        parts.append(embed_batch(client, model, chunk))
        print(f"    {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(parts)


def check(mat, n_rows):
    """取り違え・欠損を落とす。API 応答をそのまま信じない。"""
    assert mat.shape[0] == n_rows, f"行数不一致 {mat.shape[0]} != {n_rows}"
    assert np.isfinite(mat).all(), "非有限値が混入している"
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"L2ノルムが1でない (min={norms.min():.4f})"
    # 全行が同一ベクトル＝入力の取り違えを検出する
    assert np.ptp(mat, axis=0).max() > 1e-6, "全行が同一ベクトル"


def estimate_tokens(texts):
    """文字数からの粗い見積り。日本語は概ね 1トークン ≒ 1.1文字。"""
    return int(sum(len(t) for t in texts) / 1.1)


def run(project, slugs, model, batch_size, out_dir, overwrite, yes):
    targets = {s: COLUMNS[s] for s in slugs}
    loaded = {s: load_texts(project, c) for s, c in targets.items()}

    total_tokens = sum(
        estimate_tokens(tr) + estimate_tokens(te)
        for (tr, _), (te, _) in loaded.values()
    )
    price = PRICE_PER_1M.get(model)
    cost = f"${total_tokens / 1e6 * price:.3f}" if price else "不明"
    print(f"モデル: {model}")
    print(f"対象列: {list(targets.values())}")
    print(f"推定トークン: {total_tokens:,}  推定コスト: {cost}")
    if not yes:
        if input("実行する? [y/N] ").strip().lower() != "y":
            raise SystemExit("中止")

    from openai import OpenAI

    client = OpenAI(api_key=get_api_key())
    os.makedirs(os.path.join(project, out_dir), exist_ok=True)

    for slug, col in targets.items():
        path = os.path.join(project, out_dir, f"_emb_{slug}_{model}.npz")
        if os.path.exists(path) and not overwrite:
            print(f"[{slug}] 既存のためスキップ: {path}")
            continue
        print(f"[{slug}] {col} を埋め込む", flush=True)
        (tr_txt, tr_ids), (te_txt, te_ids) = loaded[slug]

        t0 = time.time()
        tr_mat = embed_all(client, model, tr_txt, batch_size)
        te_mat = embed_all(client, model, te_txt, batch_size)
        check(tr_mat, len(tr_txt))
        check(te_mat, len(te_txt))

        np.savez_compressed(
            path, train=tr_mat, test=te_mat, train_ids=tr_ids, test_ids=te_ids,
            meta=np.array([model, col, str(tr_mat.shape[1])], dtype=object),
        )
        print(f"[{slug}] 保存: {path}  形状 {tr_mat.shape} / {te_mat.shape}"
              f"  {time.time() - t0:.1f}s")

    print("\n完了。この npz をローカルに持ち帰れば、以降 API は不要。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=".", help="リポジトリのルート")
    p.add_argument("--cols", nargs="+", default=list(COLUMNS),
                   choices=list(COLUMNS), help="埋め込む列（既定は3列すべて）")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64,
                   help="1リクエストあたりの文書数。1文書≈900トークンなので64で約58k")
    p.add_argument("--out-dir", default="exp")
    p.add_argument("--overwrite", action="store_true",
                   help="既存 npz を作り直す（再現性を壊すので原則使わない）")
    p.add_argument("--yes", action="store_true", help="コスト確認を飛ばす")
    a = p.parse_args()
    sys.exit(run(a.project, a.cols, a.model, a.batch_size, a.out_dir,
                 a.overwrite, a.yes))
