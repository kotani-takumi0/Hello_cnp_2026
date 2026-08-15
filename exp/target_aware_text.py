"""購入ラベルで文章空間を更新する SetFit-style text expert。

外部の ``setfit`` / ``sentence-transformers`` パッケージへ依存せず、既に利用している
``torch`` と ``transformers`` だけで次を実装する。

1. outer-train のラベルだけから same-label / different-label pair を作る。
2. SentenceTransformer系encoderを cosine similarity loss でfine-tuneする。
3. 更新後encoderの文書embeddingに低自由度Logistic Regressionをfitする。
4. outer-validationを予測する。

これはSetFitの核である contrastive sentence-pair tuning -> classification head を再現する
最小実験である。重要なのは、encoderのfine-tuneもheadのfitも外側fold内に閉じること。
全ラベルでencoderを更新してからCVする使い方は禁止する。

DX展望は結論が末尾に置かれることが多い。通常の先頭truncationでは末尾を落とすため、
上限を超えた文書は先頭と末尾を半分ずつ残す head-tail truncation を使う。
"""
from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_PREFIX = "passage: "


@dataclass(frozen=True)
class TargetAwareConfig:
    model_name: str = DEFAULT_MODEL
    prefix: str = DEFAULT_PREFIX
    max_length: int = 128
    epochs: int = 1
    batch_size: int = 8
    encode_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_last_n_layers: int = 2
    pairs_per_anchor: int = 1
    head_c: float = 1.0
    threads: int = 4
    local_files_only: bool = True
    max_steps: int | None = None


def seed_everything(seed: int, threads: int = 4) -> None:
    """Python / NumPy / Torchの乱数とCPUスレッド数を固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _head_tail_ids(tokenizer, text: str, prefix: str,
                   max_length: int) -> dict[str, list[int]]:
    """特殊token込みでmax_length以内になるhead-tail token列を返す。"""
    width = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if width < 2:
        raise ValueError(f"max_length={max_length} は短すぎる")
    ids = tokenizer(prefix + str(text), add_special_tokens=False)["input_ids"]
    if len(ids) > width:
        left = (width + 1) // 2
        right = width - left
        ids = ids[:left] + (ids[-right:] if right else [])
    # transformers 5系では prepare_for_model / build_inputs_with_special_tokens が
    # tokenizer公開APIから削除された。今回使うBERT/RoBERTa系は単文入力が
    # [CLS/BOS] tokens [SEP/EOS] なので、IDを明示的に組み立てる。
    start = tokenizer.cls_token_id
    end = tokenizer.sep_token_id
    if start is None or end is None:
        raise ValueError(
            f"{type(tokenizer).__name__} にcls/sep tokenがないためhead-tail化できない"
        )
    input_ids = [start] + ids + [end]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def tokenize_batch(tokenizer, texts: Sequence[str], prefix: str,
                   max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    items = [_head_tail_ids(tokenizer, t, prefix, max_length) for t in texts]
    batch = tokenizer.pad(items, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def encode_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**batch).last_hidden_state
    return mean_pool(out, batch["attention_mask"])


def balanced_pairs(y: np.ndarray, seed: int,
                   pairs_per_anchor: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """class-balanced anchorから同ラベル・異ラベルpairを同数作る。

    負例が563件、正例が179件なので全行をanchorにするとnegative class内pairが支配する。
    少数class件数にanchor数を揃え、partner側では両classの全候補を利用する。
    """
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"binary label 0/1を想定: {classes}")
    by_class = {c: np.flatnonzero(y == c) for c in classes}
    if min(map(len, by_class.values())) < 2:
        raise ValueError("same-label pairを作るには各classが2件以上必要")

    rng = np.random.default_rng(seed)
    n_anchor = min(map(len, by_class.values()))
    anchors = []
    for c in classes:
        anchors.extend(rng.choice(by_class[c], size=n_anchor, replace=False))
    rng.shuffle(anchors)

    left, right, target = [], [], []
    for anchor in anchors:
        same_pool = by_class[y[anchor]]
        same_pool = same_pool[same_pool != anchor]
        diff_pool = by_class[1 - y[anchor]]
        for _ in range(pairs_per_anchor):
            left.extend((anchor, anchor))
            right.extend((rng.choice(same_pool), rng.choice(diff_pool)))
            target.extend((1.0, 0.0))

    order = rng.permutation(len(target))
    return (np.asarray(left, dtype=int)[order],
            np.asarray(right, dtype=int)[order],
            np.asarray(target, dtype=np.float32)[order])


def _encoder_layers(model):
    """BERT/RoBERTa系encoder block列を取得する。"""
    base = getattr(model, model.base_model_prefix, model)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError(
            f"{type(model).__name__} のencoder layerを特定できない。"
            "対象モデル用に _encoder_layers を追加すること。"
        )
    return layers


def freeze_except_last_layers(model, n_layers: int) -> int:
    """encoder末尾n blockだけを学習対象にし、学習parameter数を返す。"""
    for param in model.parameters():
        param.requires_grad = False
    layers = _encoder_layers(model)
    if n_layers <= 0 or n_layers > len(layers):
        raise ValueError(f"train_last_n_layers={n_layers}, model layers={len(layers)}")
    for layer in layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_encoder(config: TargetAwareConfig, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, local_files_only=config.local_files_only,
    )
    model = AutoModel.from_pretrained(
        config.model_name, local_files_only=config.local_files_only,
    ).to(device)
    n_trainable = freeze_except_last_layers(model, config.train_last_n_layers)
    return tokenizer, model, n_trainable


def contrastive_tune(model, tokenizer, texts: np.ndarray, y: np.ndarray,
                     config: TargetAwareConfig, seed: int,
                     device: torch.device) -> dict[str, float]:
    """outer-trainだけを使ってcosine pair lossでencoderを更新する。"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    losses = []
    steps = 0
    model.train()
    for epoch in range(config.epochs):
        left, right, target = balanced_pairs(
            y, seed + epoch * 1009, config.pairs_per_anchor,
        )
        for start in range(0, len(target), config.batch_size):
            if config.max_steps is not None and steps >= config.max_steps:
                break
            sl = slice(start, start + config.batch_size)
            pair_texts = np.concatenate((texts[left[sl]], texts[right[sl]]))
            batch = tokenize_batch(
                tokenizer, pair_texts, config.prefix, config.max_length, device,
            )
            z = encode_batch(model, batch)
            n = len(target[sl])
            similarity = (z[:n] * z[n:]).sum(dim=1)
            labels = torch.as_tensor(target[sl], device=device)
            loss = F.mse_loss(similarity, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            steps += 1
        if config.max_steps is not None and steps >= config.max_steps:
            break
    return {"loss": float(np.mean(losses)), "steps": steps}


def encode_texts(model, tokenizer, texts: np.ndarray, config: TargetAwareConfig,
                 device: torch.device) -> np.ndarray:
    """更新後encoderで文書embeddingを作る。"""
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), config.encode_batch_size):
            batch = tokenize_batch(
                tokenizer, texts[start:start + config.encode_batch_size],
                config.prefix, config.max_length, device,
            )
            chunks.append(encode_batch(model, batch).cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def fit_predict_fold(train_texts: Sequence[str], train_y: Sequence[int],
                     valid_texts: Sequence[str], config: TargetAwareConfig,
                     seed: int, extra_texts: Sequence[str] | None = None):
    """1 outer foldを完結させ、validationと任意extraの確率を返す。"""
    seed_everything(seed, config.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_text = np.asarray(train_texts, dtype=object)
    va_text = np.asarray(valid_texts, dtype=object)
    y = np.asarray(train_y, dtype=int)

    tokenizer, model, n_trainable = load_encoder(config, device)
    diag = contrastive_tune(model, tokenizer, tr_text, y, config, seed, device)
    tr_emb = encode_texts(model, tokenizer, tr_text, config, device)
    va_emb = encode_texts(model, tokenizer, va_text, config, device)

    head = LogisticRegression(
        C=config.head_c, max_iter=3000, random_state=0,
    ).fit(tr_emb, y)
    valid_pred = head.predict_proba(va_emb)[:, 1]
    extra_pred = None
    if extra_texts is not None and len(extra_texts):
        ex_emb = encode_texts(
            model, tokenizer, np.asarray(extra_texts, dtype=object), config, device,
        )
        extra_pred = head.predict_proba(ex_emb)[:, 1]

    diag.update({
        "trainable_parameters": int(n_trainable),
        "device": str(device),
        "n_train": len(tr_text),
        "n_valid": len(va_text),
    })
    del head, tr_emb, va_emb, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return valid_pred, extra_pred, diag
