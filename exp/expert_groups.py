"""エキスパート分割の定義（特徴量グループ × モデル）。

単一LightGBMに94列を全部入れる現行構成に対し、グループごとに専用モデルを立てて
最後に重み付きで合成する構成を検討するための土台。ここでは「どの列がどの
エキスパートに属するか」だけを定義し、学習は行わない。

方針:
  - E0 は現行の全94列モデル。アンサンブルのアンカーとして必ず残す
    （グループをまたぐ交互作用を保持し、構造的な下振れを防ぐため）。
  - E5(企業概要) は単体AUC .650 と弱く、H15としてbestに足すと悪化した実績が
    あるため今回は不採用。
  - 規模系(従業員数/事業所数/工場数/店舗数)は欠損が激しい(工場数403/店舗数603)
    ので、線形ではなく木に任せる E1 に寄せる。
  - H21 の財務比4本は E1 だけに入れる（E0 は現行94列のまま）。木は列の除算を
    再構築できないので比は効くが、E0 上での比の追加は H2/H3/H7 で3回却下されて
    いる。詳細は finance_ratio_features.py の docstring。
"""
from cross_features import ALL_CROSS_COLS
from finance_ratio_features import FINANCE_RATIO_COLS
from survey_features import ALL_DERIVED_COLS, STEP_COLS

# E0(アンカー94列)に混ぜてはいけない、特定エキスパート専用の派生列。
EXPERT_ONLY_COLS = (tuple(FINANCE_RATIO_COLS) + tuple(ALL_DERIVED_COLS)
                    + tuple(ALL_CROSS_COLS))

FINANCE_COLS = (
    "資本金", "総資産", "流動資産", "固定資産", "負債", "短期借入金", "長期借入金",
    "純資産", "自己資本", "売上", "営業利益", "経常利益", "当期純利益", "営業CF",
    "減価償却費", "運転資本変動", "投資CF", "有形固定資産変動",
    "無形固定資産変動(ソフトウェア関連)",
)
SCALE_COLS = ("従業員数", "事業所数", "工場数", "店舗数")
H1_COLS = ("営業利益率", "経常利益率", "純利益率")
FINANCE_ISNA_COLS = (
    "事業所数_isna", "工場数_isna", "店舗数_isna",
    "資本金_isna", "営業利益_isna", "経常利益_isna",
)

SURVEY_COLS = tuple(f"アンケート{n}" for n in "１２３４５６７８９") + (
    "アンケート１０", "アンケート１１",
)
SURVEY_ISNA_COLS = ("アンケート７_isna",)

CAT_COLS = ("業界", "上場種別", "特徴")


def e0_anchor_cols(all_cols):
    """E0 アンカー: 現行94列。E1専用の H21 比も E2専用の H22 派生も含めない。

    専用列を積んでいない特徴量行列に対しては全列をそのまま返す（＝exp019 と同一）。
    """
    return [c for c in all_cols if c not in EXPERT_ONLY_COLS]


def e1_finance_cols(with_ratios=False):
    """E1 財務・規模: 連続量と比率。欠損が多いので木向き。

    with_ratios=True で H21 の財務比4本を足す。既定を False にしてあるのは、
    exp019 以前の記録値をコード変更なしで再現できるようにするため。
    """
    cols = list(FINANCE_COLS + SCALE_COLS + H1_COLS + FINANCE_ISNA_COLS)
    return cols + list(FINANCE_RATIO_COLS) if with_ratios else cols


def e2_survey_cols(with_step=False):
    """E2 アンケート: 全て1〜5の順序尺度。線形が素直に効くはず。

    with_step=True で H22 の閾値指標4本を足す。E2単体の20seedペア比較で
    ΔAP +0.0151 (20/20) の ACCEPT(明確)。既定を False にしてあるのは、
    exp019 以前の記録値をコード変更なしで再現できるようにするため。
    同じ H22 の shape群/gap群は却下済み（`survey_features` の docstring 参照）。
    """
    cols = list(SURVEY_COLS + SURVEY_ISNA_COLS)
    return cols + list(STEP_COLS) if with_step else cols


def e7_cross_cols():
    """E7 不満+財務の低次元専門家12列（親7 + 積5、H27 / exp030）。

    名称は再現性のためcrossのままだが、事後アブレーションでは親7列だけで
    AP 0.6853、全12列で0.6859。成功の本体は積ではなく、E1/E2が個別には持たない
    不満集約・財務比率を低次元LRへまとめたこと。既存6本の入力列は変えない。
    詳細は cross_features.py の docstring。
    """
    return list(ALL_CROSS_COLS)


def e6_manual_cols(all_cols):
    """E6 辞書・構造: H14 / H13手作り / H18 の疎な0/1 + カテゴリ3列。

    列名の接頭辞で拾う。`組織図_購入確率` はfold内で作るスコアなのでここには
    含めない（E4が担当する）。
    """
    prefixes = ("DX展望_", "組織図_", "概要_")
    manual = [
        c for c in all_cols
        if c.startswith(prefixes) and not c.endswith("購入確率")
    ]
    return manual + list(CAT_COLS)
