"""H51: DX投資の決裁・予算化・実施時期の具体性を表す固定特徴。"""
import re
import unicodedata

import numpy as np
import pandas as pd


PROCUREMENT_TIMING_COLS = (
    "購買時点_予算明示",
    "購買時点_時期明示",
    "購買時点_決裁プロセス",
    "購買時点_実行コミット",
    "購買時点_未確定",
    "購買時点_具体性",
    "購買時点_予算x時期",
    "購買時点_実行準備完了",
)

_SPACE = re.compile(r"[\s　]+")
BUDGET_PATTERNS = (
    "予算", "予算化", "投資額", "投資枠", "資本投下", "経営資源を投入",
    "資金を投入", "重点投資", "成長投資",
)
TIMING_PATTERNS = (
    "来期", "次年度", "今年度", "年度内", "年内", "上期", "下期", "四半期",
    "中期経営計画", "中期計画", "ロードマップ", "スケジュール", "期限",
    "短期", "中期", "長期", "段階的に",
)
DECISION_PATTERNS = (
    "経営会議", "取締役会", "決裁", "承認", "稟議", "予算化", "経営判断",
    "意思決定", "経営計画", "経営戦略",
)
COMMIT_PATTERNS = (
    "着手", "開始します", "開始する", "実施します", "実施する", "導入します",
    "導入する", "展開します", "展開する", "構築します", "構築する",
    "刷新します", "刷新する", "本格化", "具体化",
)
UNCERTAIN_PATTERNS = (
    "検討", "見極め", "様子見", "必要に応じ", "慎重", "費用対効果",
    "優先順位", "可能性", "視野に入れ", "予定はない", "見送", "凍結",
)


def _normalize(value):
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return _SPACE.sub(" ", text).strip()


def _has(text, patterns):
    return int(any(pattern in text for pattern in patterns))


def make_procurement_timing_features(raw):
    rows = []
    for value in raw["今後のDX展望"]:
        text = _normalize(value)
        budget = _has(text, BUDGET_PATTERNS)
        timing = _has(text, TIMING_PATTERNS)
        decision = _has(text, DECISION_PATTERNS)
        commit = _has(text, COMMIT_PATTERNS)
        uncertain = _has(text, UNCERTAIN_PATTERNS)
        specificity = budget + timing + decision + commit
        rows.append((
            budget, timing, decision, commit, uncertain, specificity,
            budget * timing,
            int(timing and commit and (budget or decision) and not uncertain),
        ))
    return pd.DataFrame(rows, columns=PROCUREMENT_TIMING_COLS, index=raw.index,
                        dtype=np.int8)


def add_procurement_timing_features(raw, X):
    out = X.copy()
    features = make_procurement_timing_features(raw)
    for col in PROCUREMENT_TIMING_COLS:
        out[col] = features[col].to_numpy()
    return out
