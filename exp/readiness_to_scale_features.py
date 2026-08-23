"""H49: DXの現在段階と将来到達段階からReadiness-to-Scaleを作る。

段階は 0=情報収集以前, 1=計画, 2=PoC/試験, 3=部分運用,
4=全社展開, 5=定着・高度化。購入ラベルやtrain内統計は使わず、DX展望だけを
固定ルーブリックで行単位変換する。
"""
import re
import unicodedata

import numpy as np
import pandas as pd


READINESS_COLS = (
    "展開段階_現在",
    "展開段階_将来",
    "展開段階_移行幅",
    "展開段階_scale_window",
    "展開段階_既に全社展開",
    "展開段階_定着高度化期",
    "展開段階_PoCから展開",
    "展開段階_停滞縮小",
)

_SENTENCE = re.compile(r"(?<=[。！？])|[\r\n]+")
_SPACE = re.compile(r"[\s　]+")
CURRENT_MARKERS = (
    "これまで", "従来", "現状", "現在", "足元", "既に", "すでに", "実績",
    "してき", "しました", "しており", "にとどまり", "段階にとどま",
)
FUTURE_MARKERS = (
    "今後", "これから", "将来", "予定", "方針", "計画", "目指", "していく",
    "まいります", "取り組む", "進める考え", "視野に入れ",
)

# 高い段階から判定する。各段階は「その状態に到達した」ことを示す語に限定する。
STAGE_PATTERNS = {
    5: (
        "定着", "浸透", "高度化", "最適化", "自律的", "継続的改善",
        "改善サイクル", "深化", "成熟",
    ),
    4: (
        "全社展開", "全社的に展開", "全社規模", "全社横断", "全社共通",
        "横展開", "水平展開", "ロールアウト", "本格展開", "全拠点",
        "グループ全体", "標準化",
    ),
    3: (
        "一部導入", "一部の業務", "一部業務", "一部部門", "特定部門",
        "限定的な導入", "局所的", "部分的", "運用開始", "実運用",
        "導入済", "活用を開始", "成果を確認", "一定の成果",
    ),
    2: (
        "PoC", "POC", "概念実証", "実証実験", "実証", "試験導入", "試験的",
        "試験運用", "パイロット", "小規模導入", "トライアル",
    ),
    1: (
        "情報収集", "検討", "構想", "計画策定", "ロードマップ策定", "調査",
        "模索", "準備", "方針を策定",
    ),
}


def _normalize(value):
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return _SPACE.sub(" ", text).strip()


def _stage(text):
    for stage in range(5, 0, -1):
        if any(pattern in text for pattern in STAGE_PATTERNS[stage]):
            return stage
    return 0


def _temporal_parts(text):
    current, future = [], []
    mode = "current"
    for sentence in (part.strip() for part in _SENTENCE.split(text)):
        if not sentence:
            continue
        has_future = any(marker in sentence for marker in FUTURE_MARKERS)
        has_current = any(marker in sentence for marker in CURRENT_MARKERS)
        if has_future and not has_current:
            mode = "future"
        if has_current:
            current.append(sentence)
        if has_future or mode == "future":
            future.append(sentence)
        elif not has_current:
            current.append(sentence)
    # 文書全体が未来方針として書かれ、過去マーカーが無い場合の保険。
    if not future:
        future = [text]
    return " ".join(current), " ".join(future)


def make_readiness_features(raw):
    rows = []
    for value in raw["今後のDX展望"]:
        text = _normalize(value)
        current_text, future_text = _temporal_parts(text)
        current, future = _stage(current_text), _stage(future_text)
        transition = future - current
        rows.append((
            current / 5.0,
            future / 5.0,
            transition / 5.0,
            int(current in (2, 3) and future >= 4),
            int(current >= 4),
            int(current >= 4 and future >= 5),
            int(current == 2 and future >= 4),
            int(future <= current),
        ))
    return pd.DataFrame(rows, columns=READINESS_COLS, index=raw.index,
                        dtype=float)


def add_readiness_features(raw, X):
    out = X.copy()
    features = make_readiness_features(raw)
    for col in READINESS_COLS:
        out[col] = features[col].to_numpy()
    return out
