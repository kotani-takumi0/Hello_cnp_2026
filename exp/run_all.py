"""H1..H5 を各々 baseline と独立比較して一括判定。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import run_hypothesis as R

for name in ["H1", "H2", "H3", "H4", "H5"]:
    print("=" * 70)
    R.main(name)
    print()
