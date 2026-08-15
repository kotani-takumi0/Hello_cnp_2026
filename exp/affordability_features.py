"""H40: DX教育商材の対象人数に対する支払余力3列。

既存の財務派生との差分:
  - H3 はソフト投資/売上/総資産の従業員当たりで、CF・営業利益は未使用。
  - H7/H21/E7 は売上・資産を分母にした比率で、教育対象人数を分母にしない。
  - ここでは「全社員へ提供するときの負担」を表す3列だけに固定する。

いずれも行単位の決定的変換で、ラベルや全体分布を使わない。
"""
import numpy as np
import pandas as pd


AFFORDABILITY_COLS = (
    "支払余力_signed_log営業CF_per従業員",
    "支払余力_signed_log営業利益_per従業員",
    "支払余力_ソフト投資対営業CF圧力",
)


def _signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def affordability_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """3列を返す。大きいほど、最初の2列は余力、3列目は圧力が強い。"""
    employees = raw["従業員数"].astype(float).clip(lower=1.0)
    operating_cf = raw["営業CF"].astype(float)
    operating_profit = raw["営業利益"].astype(float)
    # 投資CFの表記規約に合わせ、負値で記録されたソフト投資を正の支出額へ戻す。
    software_investment = (
        -raw["無形固定資産変動(ソフトウェア関連)"].astype(float)
    ).clip(lower=0.0)

    out = pd.DataFrame(index=raw.index)
    out[AFFORDABILITY_COLS[0]] = _signed_log1p(operating_cf / employees)
    out[AFFORDABILITY_COLS[1]] = _signed_log1p(operating_profit / employees)
    # CF<=0でも欠損にせず、signed-logを引くことで資金圧力を大きく表す。
    out[AFFORDABILITY_COLS[2]] = (
        np.log1p(software_investment) - _signed_log1p(operating_cf)
    )
    return out

