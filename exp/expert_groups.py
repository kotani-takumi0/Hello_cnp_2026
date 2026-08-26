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
from survey_features import ALL_DERIVED_COLS, H57_SUBSCALE_COLS, STEP_COLS

# E0(アンカー94列)に混ぜてはいけない、特定エキスパート専用の派生列。
# 負債資産比率はh2_financial_health()が常に計算するが、既定のH2-both4
# 再現性を壊さないよう既定では除外し、--h2-stable-leverage指定時だけ
# 明示的に追加する(holdout_check.py参照)。
EXPERT_ONLY_COLS = (tuple(FINANCE_RATIO_COLS) + tuple(ALL_DERIVED_COLS)
                    + tuple(ALL_CROSS_COLS) + tuple(H57_SUBSCALE_COLS)
                    + ("負債資産比率",))

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


H2_FINANCE_HEALTH_COLS = ("自己資本比率", "負債比率", "ROA", "ROE")


# H54: target-freeなSpearman相関(signed-log変換後、train742行、yは一切未使用)で
# FINANCE_COLS(19列の生金額)を検査した結果(2026-08-25)。
#   しきい値0.85〜0.90では「企業規模」という単一の潜在因子で総資産・流動資産・
#   固定資産・負債・純資産・自己資本の6列が連鎖的に1塊へ併合されてしまい、
#   E1が自己資本と総資産の比のような情報を作れなくなる(H-E1-1が警告する
#   「amount系を1つに潰すとratio的signalが消える」に該当)。
#   しきい値0.97まで上げると連鎖が解け、会計上ほぼ同一の4組だけが残る:
#     総資産≒流動資産≒固定資産(rho>=0.991)、純資産≒自己資本(rho=0.999)、
#     営業利益≒経常利益(rho=0.993)、当期純利益≒営業CF(rho=0.997)。
#   代表列は「欠損率最小→数値安定性→解釈可能性」の順(target APでは選ばない)。
#   総資産・自己資本・当期純利益は欠損0%で他エキスパート(H2比率等)でも
#   そのまま名前が使われる標準的な指標なので優先。経常利益(欠損11.2%)より
#   営業利益(欠損5.8%)を残す。
H54_FINANCE_DROP_COLS = ("流動資産", "固定資産", "純資産", "経常利益", "営業CF")


def e1_finance_cols(with_ratios=False, with_h2=False, h54_prune=False):
    """E1 財務・規模: 連続量と比率。欠損が多いので木向き。

    with_ratios=True で H21 の財務比4本を足す。既定を False にしてあるのは、
    exp019 以前の記録値をコード変更なしで再現できるようにするため。

    with_h2=True で H2財務健全性比4本(自己資本比率/負債比率/ROA/ROE)を足す。
    2026-08-23にE0/E0bへ復元してACCEPTされた列だが、E1自体には一度も
    足されたことがない（H2はE0/E0bの構成でしか検証されていない）。
    build_features側でh2_restore=Trueにしないとこの4列自体が存在しない。

    h54_prune=True で H54_FINANCE_DROP_COLS(target-freeな会計恒等式重複5列)
    を除く。19→14列（`documents/signate_information_quality_research.md`のH54）。
    """
    cols = list(FINANCE_COLS + SCALE_COLS + H1_COLS + FINANCE_ISNA_COLS)
    if h54_prune:
        cols = [c for c in cols if c not in H54_FINANCE_DROP_COLS]
    if with_ratios:
        cols = cols + list(FINANCE_RATIO_COLS)
    if with_h2:
        cols = cols + list(H2_FINANCE_HEALTH_COLS)
    return cols


def e2_survey_cols(with_step=False, h57_subscales=False):
    """E2 アンケート: 全て1〜5の順序尺度。線形が素直に効くはず。

    with_step=True で H22 の閾値指標4本を足す。E2単体の20seedペア比較で
    ΔAP +0.0151 (20/20) の ACCEPT(明確)。既定を False にしてあるのは、
    exp019 以前の記録値をコード変更なしで再現できるようにするため。
    同じ H22 の shape群/gap群は却下済み（`survey_features` の docstring 参照）。

    h57_subscales=True で生11列(+isna)の代わりにH57の意味サブスケール4列
    だけを使う（raw置換、両立させない。`survey_features.add_h57_subscales`）。
    """
    if h57_subscales:
        return list(H57_SUBSCALE_COLS)
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


LLM_CORE4_COLS = ("LLM_need_score", "LLM_make_limitation_score",
                  "LLM_buy_openness_score",
                  "LLM_internal_training_capability_score")


# H52: target-freeなJaccard類似度(2026-08-25診断、train742行、yは一切未使用)で
# 二値フラグ28本(組織図_has_*14 + DX展望_の二値8 + 概要_の二値6。DX展望_の
# カウント/比率/差分14本は連続シグナルなので対象外)を検査した結果。
#   - 完全重複(jac=1.00): 組織図_has_dx と 組織図_has_dx推進 は同一列。
#   - ほぼ重複クラスタ(全ペアjac>=0.816): DX展望_has_cautious/has_education/
#     has_future_education/has_vendor/education_x_expand/education_x_cautious/
#     vendor_x_education/has_future_cautious の8本。支持率86.3%〜99.6%で
#     天井近くに張り付いており、実質同じ「DX展望文に何かしら前向き/vendor/教育
#     系の語がある」を8回渡している。8本中もっとも支持率が中庸(86.3%、天井から
#     最も遠い=情報量が残っている)なhas_future_cautiousだけを代表として残す。
#   - レアフラグ(support<15, train742行中): 概要_high_iot(6件)・概要_high_dx(10件)。
# 上記以外のペアはjac<0.75で「ほぼ同義」と呼べる水準になく、削らない
# （例: 組織図_has_技術開発/has_経営企画はjac=0.590で据え置き）。
H52_DROP_COLS = (
    "組織図_has_dx推進",
    "DX展望_has_cautious", "DX展望_has_education", "DX展望_has_future_education",
    "DX展望_has_vendor", "DX展望_education_x_expand", "DX展望_education_x_cautious",
    "DX展望_vendor_x_education",
    "概要_high_iot", "概要_high_dx",
)


def e6_manual_cols(all_cols, drop_dx_manual=False, add_llm4=False,
                   h52_prune=False):
    """E6 辞書・構造: H14 / H13手作り / H18 の疎な0/1 + カテゴリ3列。

    列名の接頭辞で拾う。`組織図_購入確率` はfold内で作るスコアなのでここには
    含めない（E4が担当する）。

    drop_dx_manual=True で DX展望_ 手動辞書22列を除く。E0のブロックablation
    (`result_e0_block_ablation.json`)で同じ22列がE0内ではノイズと判明し、
    LLM親4軸に置換済み（8/23）。E6単体でdrop_dx_manualだけ試すとAUC/AP/F1が
    全滅する catastrophic な悪化になった（2026-08-25、E6は他エキスパートと違い
    DX展望_がほぼ主要な情報源のため）。add_llm4=True を併用すると削除ではなく
    E0と同じLLM親4軸へ置換する形になる。既定は両方Falseで既存の記録値を再現する。

    h52_prune=True で H52_DROP_COLS(target-freeな重複/レアフラグ10本)だけを
    除く。drop_dx_manualと違い、DX展望_のカウント/比率/差分14本と二値クラスタの
    代表1本は残すので、情報を大きく削らない設計（`documents/
    signate_information_quality_research.md`のH52）。
    """
    prefixes = (("組織図_", "概要_") if drop_dx_manual
                else ("DX展望_", "組織図_", "概要_"))
    manual = [
        c for c in all_cols
        if c.startswith(prefixes) and not c.endswith("購入確率")
    ]
    if h52_prune:
        manual = [c for c in manual if c not in H52_DROP_COLS]
    extra = list(LLM_CORE4_COLS) if add_llm4 else []
    return manual + list(CAT_COLS) + extra
