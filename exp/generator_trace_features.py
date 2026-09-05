"""H90: target-free fingerprints left by synthetic-data generation.

This module deliberately stays on the ordinary feature-engineering side of
the competition rules.  It uses only values already present in each supplied
row and does not attempt to recover a pseudo-random generator, infer hidden
labels from leaderboard feedback, or query any external source.

The features describe *how* a row was rendered rather than what the company
says:

* terminal digits, divisibility and trailing-zero patterns of numeric fields;
* +/-1 rounding residuals in the two balance-sheet identities;
* low-order residues of mechanically related cash-flow fields;
* missing-value patterns; and
* character, punctuation, line and sentence layout of the four text fields.

All transforms are deterministic and row-local.  A strongly regularised
logistic-regression head is used so that the roughly two hundred weak traces
cannot be freely memorised at n=742.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRACE_EXPERT_NAME = "E13_generator_trace"
TRACE_HEAD_C = 0.01
TRACE_GROUP_PREFIXES = {
    "numeric_digits": ("numeric__", "row__", "quantised__"),
    "accounting_residuals": ("identity__", "bridge__"),
    "text_layout": ("layout__", "name__"),
    "numeric_all": (
        "numeric__", "row__", "quantised__", "identity__", "bridge__",
    ),
}

NUMERIC_TRACE_COLS = (
    "従業員数", "事業所数", "工場数", "店舗数", "資本金", "総資産",
    "流動資産", "固定資産", "負債", "短期借入金", "長期借入金", "純資産",
    "自己資本", "売上", "営業利益", "経常利益", "当期純利益", "営業CF",
    "減価償却費", "運転資本変動", "投資CF", "有形固定資産変動",
    "無形固定資産変動(ソフトウェア関連)",
)
TEXT_TRACE_COLS = ("企業名", "企業概要", "組織図", "今後のDX展望")
QUANTISED_COLS = (
    "短期借入金", "長期借入金", "減価償却費", "運転資本変動",
    "有形固定資産変動",
)

_DIGIT = re.compile(r"[0-9０-９]")
_ASCII = re.compile(r"[\x00-\x7f]")
_HIRAGANA = re.compile(r"[ぁ-ゖ]")
_KATAKANA = re.compile(r"[ァ-ヺー]")
_KANJI = re.compile(r"[一-龯々]")
_WHITESPACE = re.compile(r"[\s　]")
_BULLET = re.compile(r"[・●○■□◆◇▶▷►▸\-＊*]")
_BRACKET = re.compile(r"[()（）\[\]［］{}｛｝「」『』【】<>＜＞]")


def _require_columns(frame: pd.DataFrame, columns, group: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{group}に必要な列が無い: {missing}")


def _trailing_zeros(values: np.ndarray, cap: int = 6) -> np.ndarray:
    """Count decimal trailing zeros of absolute rounded integers.

    Zero is assigned ``cap`` rather than an unbounded count.  Missing values
    are supplied as zero by the caller and remain distinguishable through an
    explicit missing flag.
    """
    integers = np.abs(np.rint(values)).astype(np.int64, copy=False)
    output = np.zeros(len(integers), dtype=float)
    zero = integers == 0
    output[zero] = cap
    work = integers.copy()
    for level in range(1, cap + 1):
        divisible = (~zero) & (work % 10 == 0)
        output[divisible] = level
        work[divisible] //= 10
    return output


def _cyclic_remainder(values: np.ndarray, modulus: int) -> tuple[np.ndarray,
                                                                  np.ndarray]:
    remainder = np.mod(np.abs(np.rint(values)).astype(np.int64), modulus)
    angle = 2.0 * np.pi * remainder / modulus
    return np.sin(angle), np.cos(angle)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _line_statistics(text: str) -> tuple[float, float, float]:
    lines = text.splitlines() or [text]
    lengths = np.asarray([len(line) for line in lines], dtype=float)
    return float(lengths.mean()), float(lengths.std()), float(lengths.max())


def _text_layout(text: object) -> dict[str, float]:
    value = "" if pd.isna(text) else unicodedata.normalize("NFKC", str(text))
    length = len(value)
    line_mean, line_std, line_max = _line_statistics(value)
    nonempty_lines = sum(bool(line.strip()) for line in value.splitlines())
    return {
        "chars": float(length),
        "utf8_bytes": float(len(value.encode("utf-8"))),
        "newlines": float(value.count("\n")),
        "nonempty_lines": float(nonempty_lines),
        "sentences": float(sum(value.count(mark) for mark in "。！？!?")),
        "jp_commas": float(value.count("、")),
        "ascii_commas": float(value.count(",")),
        "colons": float(value.count(":") + value.count("：")),
        "semicolons": float(value.count(";") + value.count("；")),
        "slashes": float(value.count("/") + value.count("／")),
        "bullets": float(len(_BULLET.findall(value))),
        "brackets": float(len(_BRACKET.findall(value))),
        "digits": float(len(_DIGIT.findall(value))),
        "spaces": float(len(_WHITESPACE.findall(value))),
        "ascii_ratio": _safe_ratio(len(_ASCII.findall(value)), length),
        "hiragana_ratio": _safe_ratio(len(_HIRAGANA.findall(value)), length),
        "katakana_ratio": _safe_ratio(len(_KATAKANA.findall(value)), length),
        "kanji_ratio": _safe_ratio(len(_KANJI.findall(value)), length),
        "unique_char_ratio": _safe_ratio(len(set(value)), length),
        "line_length_mean": line_mean,
        "line_length_std": line_std,
        "line_length_max": line_max,
        "ends_with_stop": float(value.endswith(("。", "！", "？", ".", "!", "?"))),
    }


def _add_identity_residue(features: dict[str, np.ndarray], name: str,
                          values: np.ndarray) -> None:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    features[f"identity__{name}__raw"] = values
    for state in (-1, 0, 1):
        features[f"identity__{name}__is_{state:+d}"] = (values == state).astype(float)


def _add_low_order_residue(features: dict[str, np.ndarray], name: str,
                           values: np.ndarray) -> None:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    for modulus in (10, 100, 1000):
        sine, cosine = _cyclic_remainder(values, modulus)
        features[f"bridge__{name}__mod{modulus}_sin"] = sine
        features[f"bridge__{name}__mod{modulus}_cos"] = cosine
        rounded = np.abs(np.rint(values)).astype(np.int64)
        features[f"bridge__{name}__div{modulus}"] = (
            rounded % modulus == 0
        ).astype(float)


def build_generator_trace_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return finite, row-local H90 features in a stable column order."""
    _require_columns(frame, NUMERIC_TRACE_COLS, "numeric trace")
    _require_columns(frame, TEXT_TRACE_COLS, "text trace")
    n_rows = len(frame)
    features: dict[str, np.ndarray] = {}

    numeric_values = []
    for column in NUMERIC_TRACE_COLS:
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(raw)
        value = np.where(missing, 0.0, raw)
        rounded = np.abs(np.rint(value)).astype(np.int64)
        numeric_values.append(rounded)

        features[f"numeric__{column}__missing"] = missing.astype(float)
        features[f"numeric__{column}__is_zero"] = ((~missing) & (rounded == 0)).astype(float)
        features[f"numeric__{column}__trailing_zeros"] = _trailing_zeros(value)
        features[f"numeric__{column}__parity"] = (rounded % 2).astype(float)
        mod10_sin, mod10_cos = _cyclic_remainder(value, 10)
        features[f"numeric__{column}__mod10_sin"] = mod10_sin
        features[f"numeric__{column}__mod10_cos"] = mod10_cos

    numeric_matrix = np.column_stack(numeric_values)
    features["row__numeric_missing_count"] = frame[list(NUMERIC_TRACE_COLS)].isna().sum(axis=1).to_numpy(dtype=float)
    features["row__numeric_zero_count"] = (numeric_matrix == 0).sum(axis=1).astype(float)
    for modulus in (2, 5, 10, 100, 1000):
        features[f"row__divisible_{modulus}_count"] = (
            numeric_matrix % modulus == 0
        ).sum(axis=1).astype(float)
    last_digit = numeric_matrix % 10
    for digit in range(10):
        features[f"row__last_digit_{digit}_count"] = (
            last_digit == digit
        ).sum(axis=1).astype(float)

    for column in QUANTISED_COLS:
        value = np.nan_to_num(pd.to_numeric(
            frame[column], errors="coerce").to_numpy(dtype=float), nan=0.0)
        rounded = np.abs(np.rint(value)).astype(np.int64)
        for modulus in (100, 1000, 10000):
            features[f"quantised__{column}__div{modulus}"] = (
                rounded % modulus == 0
            ).astype(float)

    _add_identity_residue(
        features,
        "assets_parts",
        frame["総資産"].to_numpy(dtype=float)
        - frame["流動資産"].to_numpy(dtype=float)
        - frame["固定資産"].to_numpy(dtype=float),
    )
    _add_identity_residue(
        features,
        "assets_funding",
        frame["総資産"].to_numpy(dtype=float)
        - frame["負債"].to_numpy(dtype=float)
        - frame["純資産"].to_numpy(dtype=float),
    )
    _add_low_order_residue(
        features,
        "operating_cf",
        frame["営業CF"].to_numpy(dtype=float)
        - frame["当期純利益"].to_numpy(dtype=float)
        - frame["減価償却費"].to_numpy(dtype=float)
        - frame["運転資本変動"].to_numpy(dtype=float),
    )
    _add_low_order_residue(
        features,
        "investing_cf",
        frame["投資CF"].to_numpy(dtype=float)
        - frame["有形固定資産変動"].to_numpy(dtype=float)
        - frame["無形固定資産変動(ソフトウェア関連)"].to_numpy(dtype=float),
    )

    for column in TEXT_TRACE_COLS:
        layouts = [_text_layout(value) for value in frame[column]]
        for statistic in layouts[0]:
            features[f"layout__{column}__{statistic}"] = np.asarray(
                [row[statistic] for row in layouts], dtype=float)

    names = frame["企業名"].fillna("").astype(str)
    features["name__corporation_prefix"] = names.str.startswith("株式会社").to_numpy(dtype=float)
    features["name__corporation_suffix"] = names.str.endswith("株式会社").to_numpy(dtype=float)
    features["name__holdings"] = names.str.contains(
        "ホールディングス", regex=False).to_numpy(dtype=float)
    features["name__latin"] = names.str.contains(
        r"[A-Za-zＡ-Ｚａ-ｚ]", regex=True).to_numpy(dtype=float)

    output = pd.DataFrame(features, index=frame.index, dtype=float)
    if len(output) != n_rows or output.columns.duplicated().any():
        raise RuntimeError("generator trace feature shape is invalid")
    if not np.isfinite(output.to_numpy()).all():
        bad = output.columns[~np.isfinite(output.to_numpy()).all(axis=0)].tolist()
        raise RuntimeError(f"generator trace produced non-finite values: {bad}")
    return output


def build_trace_model(C: float = TRACE_HEAD_C):
    """Create the fixed, strongly regularised H90 head."""
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        VarianceThreshold(threshold=1e-12),
        StandardScaler(),
        LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        ),
    )


def select_trace_feature_group(trace: pd.DataFrame,
                               group: str = "all") -> pd.DataFrame:
    """Select one predeclared H90 family without looking at labels."""
    if group == "all":
        return trace.copy()
    if group not in TRACE_GROUP_PREFIXES:
        raise ValueError(
            f"unknown trace group {group!r}; "
            f"expected one of {('all', *TRACE_GROUP_PREFIXES)}"
        )
    prefixes = TRACE_GROUP_PREFIXES[group]
    columns = [column for column in trace if column.startswith(prefixes)]
    if not columns:
        raise RuntimeError(f"trace group {group!r} selected no columns")
    return trace[columns].copy()


def fit_predict_trace(trace: pd.DataFrame, y: np.ndarray, train_indices,
                      target_indices, C: float = TRACE_HEAD_C) -> np.ndarray:
    """Fit only on ``train_indices`` and predict ``target_indices``."""
    train_indices = np.asarray(train_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    labels = np.asarray(y, dtype=int)
    model = build_trace_model(C=C).fit(
        trace.iloc[train_indices], labels[train_indices])
    # Accelerate BLAS on macOS can leave harmless floating-point status flags
    # after a finite matrix product.  Suppress those flags locally, then make
    # the real safety check explicit below.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        probability = model.predict_proba(trace.iloc[target_indices])[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("generator trace head produced non-finite probabilities")
    return probability
