"""外部連携の有無ではなく、その理由・状態を文章から固定辞書で抽出する。"""
import re
import unicodedata

import numpy as np
import pandas as pd


COLS = (
    "外部理由_内製自走優先",
    "外部理由_外注拒否限定",
    "外部理由_探索検討中",
    "外部理由_既存委託協業",
    "外部理由_教育外部利用",
    "外部理由_外部計画",
)
_SPLIT = re.compile(r"(?<=[。！？])|[\r\n]+")
_SPACE = re.compile(r"[\s　]+")
EXTERNAL = ("外部", "社外", "ベンダー", "パートナー", "コンサル", "専門家",
            "委託", "外注", "アライアンス", "スタートアップ", "大学")
EDUCATION = ("研修", "教育", "講座", "セミナー", "eラーニング", "リスキリング",
             "育成", "トレーニング", "学習")
INTERNAL = ("内製", "自社内", "社内で", "自前", "自走", "OJT", "社内勉強会",
            "社内アカデミー", "内部人材", "社内人材", "自律的")
REJECT = ("見送", "抑制", "限定", "慎重", "依存せず", "依存しない", "最小限",
          "必要に応じ", "当面行わ", "予定はない", "予定はありません")
SEARCH = ("情報収集", "探索", "選定", "比較", "検討", "模索", "候補", "視野に入")
EXISTING = ("導入済", "活用して", "利用して", "連携して", "協業して", "委託して",
            "実績", "これまで", "従来", "継続", "既存")
PLAN = ("導入する", "活用する", "連携する", "協業する", "委託する", "強化する",
        "拡大する", "取り入れる", "予定", "計画", "方針", "今後")


def _norm(value):
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return _SPACE.sub(" ", text)


def _has(sentence, terms):
    return any(term in sentence for term in terms)


def _row(text):
    sentences = [s.strip() for s in _SPLIT.split(_norm(text)) if s.strip()]
    internal = any(_has(s, INTERNAL) for s in sentences)
    reject = any(_has(s, EXTERNAL) and _has(s, REJECT) for s in sentences)
    search = any(_has(s, EXTERNAL) and _has(s, SEARCH) for s in sentences)
    existing = any(_has(s, EXTERNAL) and _has(s, EXISTING) and not _has(s, REJECT)
                   for s in sentences)
    education = any(_has(s, EXTERNAL) and _has(s, EDUCATION) and not _has(s, REJECT)
                    for s in sentences)
    plan = any(_has(s, EXTERNAL) and _has(s, PLAN) and not _has(s, REJECT)
               for s in sentences)
    return internal, reject, search, existing, education, plan


def make_external_sourcing_reason_features(raw):
    # 方針が詳しいDX展望と、内製体制が現れる組織図を結合する。
    text = raw["今後のDX展望"].fillna("") + "。" + raw["組織図"].fillna("")
    return pd.DataFrame([_row(v) for v in text], columns=COLS,
                        index=raw.index, dtype=np.int8)
