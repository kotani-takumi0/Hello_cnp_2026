"""日本語BERT系エンコーダでテキスト3列をベクトル化して npz に落とす（ローカル実行）。

`colab_openai_embed.py` の兄弟。**出力の契約（npz のキー・行順・L2正規化）を完全に
揃えてある**ので、`exp/embedding_features.py` 以下の配管は `--model` を差し替えるだけで
そのまま動く。テキストの正規化と行順は colab_openai_embed から**関数ごと import** して
共有している（別実装にすると「エンコーダの差」と「前処理の差」が交絡する）。

なぜローカルCPUで回すか:
  train 742 + test 800 = 1542文書しかない。base級なら数分、large級でも十数分で終わる。
  API課金もColabも要らないので、`--overwrite` せず一度作った npz を真実にする運用は
  OpenAI 側と同じにできる。

**OpenAI埋め込みとの3つの構造的な違い**（ここを間違えると静かに壊れる）:

  1. **Matryoshka ではない**。text-embedding-3 系は「先頭k次元を取ってL2再正規化」で
     k次元モデルになるが、BERT系の次元にその性質は無い。先頭512次元を取る操作は
     ただの情報の切り捨てになる。そのため npz の meta に `mrl=0` を書き込み、
     `embedding_features.load_embeddings` 側で切り詰めを拒否させる。
     （ruri-v3 のように MRL 学習済みのものだけ `mrl=1`）

  2. **系列長の上限がある**。今後のDX展望は平均852文字で、512トークン上限の
     モデルでは末尾が落ちる。ここでは**チャンク分割 + トークン数重み付き平均**で
     全文を読む（`--no-chunk` で単純truncationにもできる。比較用）。

  3. **プーリングとプレフィックスがモデルごとに違う**。e5 は "passage: "、ruri は
     "検索文書: " を付けないと学習時の分布から外れる。プーリングは
     sentence-transformers の `1_Pooling/config.json` がリポジトリにあればそれに従い、
     無ければ mean にフォールバックする（素のMLM BERTは mean が定石）。

使い方:
  pip install -r requirements-optional.txt   # torch(CPU) / transformers / fugashi
  python3 local_bert_embed.py --model tohoku-bert-v3
  python3 local_bert_embed.py --model ruri-v3-310m --cols org overview

出力: data/_emb_{slug}_{model}.npz
  train (n_train, d) / test (n_test, d) / train_ids / test_ids / meta
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_openai_embed import COLUMNS, check, load_texts  # noqa: E402

# モデル登録簿。key が npz のファイル名に入る短縮名。
#   hf      : HuggingFace のリポジトリID
#   prefix  : 学習時に文書側へ付けられていた接頭辞（付け忘れると分布が外れる）
#   max_len : 位置埋め込みの上限。チャンク幅の決定に使う
#   mrl     : 先頭k次元の切り詰めが許されるか（Matryoshka学習済みか）
MODELS = {
    # 素のMLM BERT。「BERT」の直球の基準線。文ベクトル用に学習されていないので
    # 弱い可能性が高いが、その差自体が「対照学習の寄与」の測定になる。
    "tohoku-bert-v3": dict(
        hf="tohoku-nlp/bert-base-japanese-v3", prefix="", max_len=512, mrl=False),
    # 多言語対照学習。日本語タスクで広く使われる実用線。
    "e5-large": dict(
        hf="intfloat/multilingual-e5-large", prefix="passage: ", max_len=512, mrl=False),
    "e5-base": dict(
        hf="intfloat/multilingual-e5-base", prefix="passage: ", max_len=512, mrl=False),
    # 日本語特化の埋め込み。ModernBERT-Ja ベースで長文をチャンク無しで読める。
    "ruri-v3-310m": dict(
        hf="cl-nagoya/ruri-v3-310m", prefix="検索文書: ", max_len=8192, mrl=True),
}

OUT_DIR_DEFAULT = "data"


def resolve_pooling(hf_id):
    """sentence-transformers の `1_Pooling/config.json` があればそれに従う。

    プーリングの取り違え（CLSのモデルをmeanで読む等）は静かに性能だけ落とす。
    モデル側が答えを持っているなら推測しない。
    """
    try:
        from huggingface_hub import hf_hub_download
        import json

        p = hf_hub_download(hf_id, "1_Pooling/config.json")
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        for mode in ("cls", "mean", "max"):
            if cfg.get(f"pooling_mode_{mode}_tokens"):
                return mode
        if cfg.get("pooling_mode_cls_token"):
            return "cls"
        if cfg.get("pooling_mode_mean_tokens"):
            return "mean"
    except Exception:
        pass
    return "mean"


def chunk_ids(ids, width, stride):
    """トークン列を width ごとに stride 刻みで切る。最低1チャンクは必ず返す。"""
    if len(ids) <= width:
        return [ids]
    out = []
    for start in range(0, len(ids), stride):
        piece = ids[start:start + width]
        if not piece:
            break
        out.append(piece)
        if start + width >= len(ids):
            break
    return out


def encode(texts, hf_id, prefix, max_len, pooling, batch_size, use_chunk, threads):
    """文書リストを (n, d) の L2正規化済み行列にする。

    長文は width=max_len-2 のチャンクに割り、チャンクのプーリング結果を
    **トークン数で重み付き平均**してから正規化する（単純平均だと末尾の
    短い断片が本文と同じ重みを持ってしまう）。
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).eval()

    width = min(max_len, tok.model_max_length or max_len) - 2
    # (doc_index, ids) の平坦なリストにしてからバッチを組む。長さ順に並べると
    # パディングが減るが、行順の取り違えを避けるため素直に元順で回す。
    units = []
    for i, t in enumerate(texts):
        ids = tok(prefix + t, add_special_tokens=False)["input_ids"]
        pieces = [ids[:width]] if not use_chunk else chunk_ids(ids, width, max(1, width * 3 // 4))
        units.extend((i, p) for p in pieces)

    dim = model.config.hidden_size
    acc = np.zeros((len(texts), dim), dtype=np.float64)
    wsum = np.zeros(len(texts), dtype=np.float64)

    with torch.inference_mode():
        for s in range(0, len(units), batch_size):
            batch = units[s:s + batch_size]
            enc = tok.pad(
                [tok.prepare_for_model(ids, add_special_tokens=True) for _, ids in batch],
                return_tensors="pt", padding=True,
            )
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            if pooling == "cls":
                vec = out[:, 0]
            else:
                vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = vec.to(torch.float64).numpy()
            for (doc, ids), v in zip(batch, vec):
                w = len(ids)
                acc[doc] += v * w
                wsum[doc] += w
            if (s // batch_size) % 20 == 0:
                print(f"    {min(s + batch_size, len(units))}/{len(units)} chunks", flush=True)

    assert (wsum > 0).all(), "空文書がある"
    mat = acc / wsum[:, None]
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat.astype(np.float32)


def run(project, slugs, model_key, batch_size, out_dir, overwrite, use_chunk, threads):
    spec = MODELS[model_key]
    pooling = resolve_pooling(spec["hf"])
    print(f"モデル: {model_key} ({spec['hf']})  pooling={pooling}  "
          f"prefix={spec['prefix']!r}  max_len={spec['max_len']}  chunk={use_chunk}")

    os.makedirs(os.path.join(project, out_dir), exist_ok=True)
    for slug in slugs:
        col = COLUMNS[slug]
        path = os.path.join(project, out_dir, f"_emb_{slug}_{model_key}.npz")
        if os.path.exists(path) and not overwrite:
            print(f"[{slug}] 既存のためスキップ: {path}")
            continue
        (tr_txt, tr_ids), (te_txt, te_ids) = load_texts(project, col)
        print(f"[{slug}] {col}  train {len(tr_txt)} / test {len(te_txt)}", flush=True)

        t0 = time.time()
        mats = [
            encode(txt, spec["hf"], spec["prefix"], spec["max_len"], pooling,
                   batch_size, use_chunk, threads)
            for txt in (tr_txt, te_txt)
        ]
        check(mats[0], len(tr_txt))
        check(mats[1], len(te_txt))

        np.savez_compressed(
            path, train=mats[0], test=mats[1], train_ids=tr_ids, test_ids=te_ids,
            meta=np.array([spec["hf"], col, str(mats[0].shape[1]),
                           f"mrl={int(spec['mrl'])}", f"pooling={pooling}",
                           f"chunk={int(use_chunk)}"], dtype=object),
        )
        print(f"[{slug}] 保存: {path}  形状 {mats[0].shape} / {mats[1].shape}"
              f"  {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=".")
    p.add_argument("--cols", nargs="+", default=list(COLUMNS), choices=list(COLUMNS))
    p.add_argument("--model", default="tohoku-bert-v3", choices=list(MODELS))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--overwrite", action="store_true",
                   help="既存 npz を作り直す（再現性を壊すので原則使わない）")
    p.add_argument("--no-chunk", action="store_true",
                   help="長文をチャンク分割せず先頭だけ読む（切り捨ての影響を見る比較用）")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SIGNATE_TORCH_THREADS", "8")))
    a = p.parse_args()
    sys.exit(run(a.project, a.cols, a.model, a.batch_size, a.out_dir,
                 a.overwrite, not a.no_chunk, a.threads))
