"""H34: 収益力(E1) × DX投資意欲(E6) のグループ横断交互作用。

E1 は財務だけ、E6 は DX展望の辞書特徴だけを学習するため、両者の積はどちらも
構造的に作れない。E0 は親列を両方持つが、木が連続量の積を再構築するには多数の
軸平行分割が必要になる。H27/E7 と同じ設計で、親列と積を専用エキスパートへ渡す。

探索時に強かった個別ペアだけをつまみ食いせず、事前に意味を固定した小ブロック:

  収益力4軸
    営業利益率 / 経常利益率 / 純利益率 / 営業CFマージン
  DX意欲2軸
    展望全文、および将来文だけの「積極語数 - 慎重語数」
  交互作用
    4 x 2 = 8本

親6列も併記する。積だけでは「収益力が高い」のか「DX意欲が高い」のかをモデルが
分離できないためで、E7 の構成と同じ。すべて行単位の決定的変換でリークはない。
"""
import numpy as np


PROFIT_COLS = (
    "営業利益率",
    "経常利益率",
    "純利益率",
    "営業CFマージン",
)
DX_INTENT_COLS = (
    "DX展望_expand_minus_cautious",
    "DX展望_future_expand_minus_cautious",
)
CROSS_COLS = tuple(f"{a}x{b}" for a in PROFIT_COLS for b in DX_INTENT_COLS)
ALL_PROFIT_DX_COLS = PROFIT_COLS + DX_INTENT_COLS + CROSS_COLS


def _safe_div(a, b):
    return (a.astype(float) / b.astype(float)).replace(
        [np.inf, -np.inf], np.nan
    )


def add_profit_dx_cross_features(raw, X):
    """生データと既存特徴行列から H34 の14列を追加する。"""
    out = X.copy()
    sales = raw["売上"]
    out["営業利益率"] = _safe_div(raw["営業利益"], sales)
    out["経常利益率"] = _safe_div(raw["経常利益"], sales)
    out["純利益率"] = _safe_div(raw["当期純利益"], sales)
    out["営業CFマージン"] = _safe_div(raw["営業CF"], sales)

    missing = [c for c in DX_INTENT_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"DX展望の親特徴が未生成: {missing}")

    for a in PROFIT_COLS:
        for b in DX_INTENT_COLS:
            out[f"{a}x{b}"] = np.asarray(
                out[a].astype(float) * out[b].astype(float), dtype=float
            )
    return out
