"""H47: DX教育商材の「購買主体の完成度」を表す固定特徴。

ドメイン仮説:
  DX教育商材の購入には、方針・予算を持つ経営企画、技術課題を持つDX/IT主体、
  研修を実施する人事教育主体の3者が必要である。単なるDX意欲ではなく、組織図上で
  buying center が成立している企業ほど、購入を実行しやすい。

設計上の制約:
  - 購入ラベルやtrain内集計を使わない行単位の決定的変換。
  - 情報源は組織図だけ。企業概要・DX展望を混ぜず、既存E3/E7の再表現を避ける。
  - 辞書と組合せは評価前に固定し、単語ごとの購入率で取捨選択しない。
"""
import re
import unicodedata

import numpy as np
import pandas as pd


BUYING_CENTER_COLS = (
    "購買主体_経営企画",
    "購買主体_DX責任",
    "購買主体_人事教育",
    "購買主体_完成度",
    "購買主体_DXx人事教育",
    "購買主体_経営xDX",
    "購買主体_三者連携",
)

_SPACE = re.compile(r"[\s　]+")

# 「取締役会」等のほぼ全社にある統治語ではなく、変革の方針・予算化を担う部署語に限定。
EXECUTIVE_PATTERNS = (
    "経営企画", "経営戦略", "戦略企画", "事業戦略", "社長室",
    "CEOオフィス", "コーポレート企画",
)
DX_OWNER_PATTERNS = (
    "DX推進", "DX戦略", "デジタル推進", "デジタル戦略", "デジタル変革",
    "IT戦略", "IT企画", "情報システム", "システム企画", "データ戦略",
)
HR_TRAINING_PATTERNS = (
    "人事", "人材開発", "人材育成", "教育研修", "研修", "能力開発",
    "組織開発", "ラーニング", "社内大学", "アカデミー",
)


def _normalize(value):
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return _SPACE.sub(" ", text).strip()


def _has_any(text, patterns):
    return int(any(pattern in text for pattern in patterns))


def make_buying_center_features(raw):
    """組織図から7個の固定特徴を作る。"""
    rows = []
    for value in raw["組織図"]:
        text = _normalize(value)
        executive = _has_any(text, EXECUTIVE_PATTERNS)
        dx_owner = _has_any(text, DX_OWNER_PATTERNS)
        hr_training = _has_any(text, HR_TRAINING_PATTERNS)
        rows.append((
            executive,
            dx_owner,
            hr_training,
            executive + dx_owner + hr_training,
            dx_owner * hr_training,
            executive * dx_owner,
            executive * dx_owner * hr_training,
        ))
    return pd.DataFrame(rows, columns=BUYING_CENTER_COLS, index=raw.index,
                        dtype=np.int8)


def add_buying_center_features(raw, X):
    out = X.copy()
    features = make_buying_center_features(raw)
    for col in BUYING_CENTER_COLS:
        out[col] = features[col].to_numpy()
    return out
