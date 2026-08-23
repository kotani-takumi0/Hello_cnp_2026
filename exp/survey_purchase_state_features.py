"""アンケート回答間の購買状態を表す、ラベル非依存の四象限特徴。"""
import numpy as np

SURVEY_PURCHASE_STATE_COLS = (
    "状態_高意欲_低実行",
    "状態_導入済み_低満足",
    "状態_高課題_高外部連携",
    "状態_低内製_高外部調達",
)


def add_survey_purchase_states(raw, frame):
    """事前固定した4状態を0/1で追加する。

    Q1/Q9/Q11: 方針・探索意欲、Q3/Q5/Q8: 内部実行・成果、
    Q2/Q7: 満足、Q4: 抵抗、Q10: 外部連携として扱う。
    Q7はQ6が導入済みの行だけ評価する。
    """
    out = frame.copy()
    numerals = ("１", "２", "３", "４", "５", "６", "７", "８", "９", "１０", "１１")
    q = {i: raw[f"アンケート{n}"].astype(float)
         for i, n in enumerate(numerals, 1)}

    intent = raw[["アンケート１", "アンケート９", "アンケート１１"]].mean(axis=1)
    internal = raw[["アンケート３", "アンケート５", "アンケート８"]].mean(axis=1)
    external = (2 * q[10] + q[9] + q[11]) / 4.0

    out[SURVEY_PURCHASE_STATE_COLS[0]] = ((intent >= 4.0) & (internal <= 3.0)).astype(np.int8)
    out[SURVEY_PURCHASE_STATE_COLS[1]] = (
        (q[6] == 1) & q[7].notna() & (q[7] <= 2)
    ).astype(np.int8)
    high_problem = (q[2] <= 2) | (q[4] >= 4)
    out[SURVEY_PURCHASE_STATE_COLS[2]] = (high_problem & (q[10] >= 4)).astype(np.int8)
    out[SURVEY_PURCHASE_STATE_COLS[3]] = ((internal <= 3.0) & (external >= 4.0)).astype(np.int8)
    return out
