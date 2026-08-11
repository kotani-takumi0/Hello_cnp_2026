"""再現性の受け入れテスト。チーム全員が最初に一度これを通すこと。

なぜ必要か
----------
このパイプラインは「確率の1e-6の揺れ」が提出ラベルに増幅される構造を持つ。
E3(DX展望テキスト)のLRスコアは下流LightGBMで gain 単独首位（2位の2.3倍）なので、
微差が 分割点 → OOF閾値 → 境界行のラベル と伝播し、**再実行するだけで提出が2行変わる**
事故が実際に起きた（liblinear の random_state 未指定が原因、修正済み）。

その2行は exp013→exp017 の Public差 +0.0254 の実体（TP 48→50 の2行）と同じ大きさだった。
つまり再現性が無いと、どの比較実験もΔが再現せず、CV設計を統一しても意味がない。

使い方
------
    python3 exp/repro_check.py --fast     # 環境 + 決定性のみ（数十秒）
    python3 exp/repro_check.py            # 上記 + 本命exp026の参照値照合（数分）
    python3 exp/repro_check.py --write-reference   # 参照値を作り直す（原則メンテナのみ）

判定
----
    [OK]   そのまま実験を始めてよい
    [WARN] 動くが記録と厳密には揃っていない。過去のOOF記録と直接比較しないこと
    [NG]   結果を共有してはいけない。原因を潰してから実験すること
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "exp", "repro_reference.npz")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")

# 本命 exp026 の記録値（documents/experiments.md）。表示桁の4桁で照合する。
REF_SEED = 42
REF_METRICS = dict(auc=0.9500, ap=0.8646, f1=0.7864, th=0.2350)
METRIC_TOL = 5e-5          # 4桁表示が一致する範囲
DRIFT_WARN = 1e-9          # OOF確率のずれ。これ以下なら実質ビット一致
DRIFT_NG = 1e-6            # 提出ラベルを動かした実績のあるオーダー


def _hdr(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def _verdict(ok, warn, msg):
    tag = "[NG]  " if not ok else ("[WARN]" if warn else "[OK]  ")
    print(f"{tag} {msg}")
    return "NG" if not ok else ("WARN" if warn else "OK")


# --------------------------------------------------------------------------
# 1. 環境
# --------------------------------------------------------------------------
def _parse_pins(path):
    """requirements.txt から `pkg==ver` のみ拾う。"""
    pins = {}
    if not os.path.exists(path):
        return pins
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "==" in line:
                name, ver = line.split("==", 1)
                pins[name.strip().lower()] = ver.strip()
    return pins


def _installed():
    try:
        from importlib.metadata import distributions
    except ImportError:  # pragma: no cover - Python 3.7 以下
        return {}
    return {d.metadata["Name"].lower(): d.version
            for d in distributions() if d.metadata["Name"]}


def check_env():
    _hdr("1. 環境")
    print(f"Python {sys.version.split()[0]}")
    pins, have = _parse_pins(REQUIREMENTS), _installed()
    if not pins:
        return _verdict(True, True, f"{REQUIREMENTS} が読めない。バージョン照合をスキップ")

    bad = []
    for name, want in sorted(pins.items()):
        got = have.get(name)
        mark = "OK" if got == want else "  "
        print(f"  {mark:2s} {name:16s} 要求 {want:10s} 実際 {got or '(未インストール)'}")
        if got != want:
            bad.append(name)

    # BLASのスレッド数は数値誤差の再現性に効きうるので記録だけ残す
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"  -- {var} = {os.environ.get(var, '(未設定)')}")

    if bad:
        return _verdict(True, True,
                        f"バージョン不一致 {len(bad)}件: {', '.join(bad)} "
                        f"→ pip install -r requirements.txt で揃えること")
    return _verdict(True, False, "requirements.txt と完全一致")


# --------------------------------------------------------------------------
# 2. 決定性（同一プロセス内で2回fitして完全一致するか）
# --------------------------------------------------------------------------
def _fit_twice(build, X, y, Xp):
    a = build().fit(X, y).predict_proba(Xp)[:, 1]
    b = build().fit(X, y).predict_proba(Xp)[:, 1]
    return float(np.abs(a - b).max())


def check_determinism():
    """random_state 未指定を直接検出する。過去にここで2行の提出差が生まれた。"""
    _hdr("2. 決定性（同一入力を2回学習して確率が完全一致するか）")
    from ensemble_experts import e4_inputs
    from text_features import TEXT_COL, build_model

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = train["購入フラグ"].values
    txt = train[TEXT_COL].fillna("").astype(str).values

    cases = [("E3 DX展望 TF-IDF+LR", lambda: _fit_twice(build_model, txt, y, txt))]
    e4, _, e4_model = e4_inputs(train, test, org_embed=True)
    cases.append(("E4 組織図 embedding+LR",
                  lambda: _fit_twice(e4_model, e4, y, e4)))

    worst = 0.0
    for name, fn in cases:
        d = fn()
        worst = max(worst, d)
        print(f"  {name:26s} 2回の確率の最大差 {d:.3e}"
              f"  {'一致' if d == 0.0 else '★不一致'}")

    if worst > 0.0:
        return _verdict(False, False,
                        "同一入力で結果が変わる。乱数が固定されていない推定器がある "
                        "（liblinear の dual は random_state 必須）")
    return _verdict(True, False, "完全一致。乱数は固定されている")


# --------------------------------------------------------------------------
# 3. 本命 exp026 の参照値照合
# --------------------------------------------------------------------------
def _compute_exp026():
    """exp026（6エキスパート + 組織図embedding + alpha自動）を再計算する。

    注意: cache=False でも exp/_experts_seed42_orgemb.npz は上書きされる
    （生成物なので問題ない。Git管理外）。
    """
    from ensemble_experts import (EXPERTS, _scores, blend_oof,
                                  compute_expert_preds)
    y, oof, te = compute_expert_preds(REF_SEED, cache=False, org_embed=True)
    blend, blend_te, w, _ = blend_oof(y, oof, te, REF_SEED, alpha=None)
    return y, oof, blend, blend_te, _scores(y, blend), w


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64)
                          .tobytes()).hexdigest()[:16]


def write_reference():
    _hdr("参照値の生成")
    y, oof, blend, blend_te, sc, _ = _compute_exp026()
    np.savez(REFERENCE, y=y, blend=blend, blend_te=blend_te,
             **{f"oof_{n}": v for n, v in oof.items()})
    print(f"保存: {REFERENCE}")
    print(f"  metrics: " + "  ".join(f"{k}={sc[k]:.6f}" for k in sc))
    print("  ※ documents/experiments.md の exp026 と一致することを必ず目視確認すること")


def check_reference():
    _hdr("3. 本命 exp026 の再現（seed=42 / 6エキスパート / 組織図embedding）")
    if not os.path.exists(REFERENCE):
        return _verdict(True, True,
                        f"参照ファイルが無い: {REFERENCE} "
                        "→ メンテナが --write-reference で作ること")

    y, oof, blend, blend_te, sc, w = _compute_exp026()
    ref = np.load(REFERENCE)

    print("\n  --- 指標（記録値との差）---")
    metric_ng = []
    for k, want in REF_METRICS.items():
        got, d = sc[k], sc[k] - REF_METRICS[k]
        ok = abs(d) <= METRIC_TOL
        print(f"    {k.upper():4s} 記録 {want:.4f}  今回 {got:.4f}  Δ {d:+.5f}"
              f"  {'' if ok else '★不一致'}")
        if not ok:
            metric_ng.append(k)

    print("\n  --- エキスパートOOF確率のずれ（参照との最大絶対差）---")
    worst = 0.0
    for n in sorted(oof):
        key = f"oof_{n}"
        if key not in ref:
            continue
        d = float(np.abs(oof[n] - ref[key]).max())
        worst = max(worst, d)
        print(f"    {n:12s} max|Δ| {d:.3e}  (hash {_digest(oof[n])}"
              f" / 参照 {_digest(ref[key])})")
    d_blend = float(np.abs(blend - ref["blend"]).max())
    worst = max(worst, d_blend)
    print(f"    {'合成OOF':12s} max|Δ| {d_blend:.3e}")

    # 実務上の最終判定: 提出ラベルが同じかどうか
    th = sc["th"]
    lab = (blend_te >= th).astype(int)
    lab_ref = (ref["blend_te"] >= REF_METRICS["th"]).astype(int)
    n_diff = int((lab != lab_ref).sum())
    print(f"\n  --- 提出ラベル ---")
    print(f"    正例数 今回 {int(lab.sum())} / 参照 {int(lab_ref.sum())}"
          f"  差分 {n_diff} 行")

    if metric_ng or n_diff > 0 or worst > DRIFT_NG:
        return _verdict(False, False,
                        f"本命が再現しない（指標不一致 {metric_ng} / ラベル差 {n_diff}行 "
                        f"/ 最大ずれ {worst:.1e}）。この環境の結果を共有しないこと")
    if worst > DRIFT_WARN:
        return _verdict(True, True,
                        f"ラベルは一致するが確率に {worst:.1e} のずれがある。"
                        "微差（ΔAP 0.0005級）の判定はこの環境では信用しないこと")
    return _verdict(True, False, "参照とビット一致。記録値と直接比較してよい")


def main():
    p = argparse.ArgumentParser(description="再現性の受け入れテスト")
    p.add_argument("--fast", action="store_true",
                   help="環境と決定性のみ（exp026の再計算をしない）")
    p.add_argument("--write-reference", action="store_true",
                   help="参照ファイルを作り直す（メンテナのみ）")
    a = p.parse_args()

    os.chdir(ROOT)  # スクリプトが相対パスで data/ を読むため
    if a.write_reference:
        write_reference()
        return 0

    results = [("環境", check_env()), ("決定性", check_determinism())]
    if not a.fast:
        results.append(("本命の再現", check_reference()))
    else:
        print("\n(--fast のため exp026 の照合をスキップ。"
              "提出物を作る前には必ずフル実行すること)")

    _hdr("判定")
    for name, v in results:
        print(f"  {v:5s}  {name}")
    if any(v == "NG" for _, v in results):
        print("\n=> NG。原因を潰すまで実験結果を共有しないこと。")
        return 1
    if any(v == "WARN" for _, v in results):
        print("\n=> WARN。動くが記録と厳密には揃っていない。"
              "過去のOOF記録との直接比較は避けること。")
        return 0
    print("\n=> OK。実験を始めてよい。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
