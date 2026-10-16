# -*- coding: utf-8 -*-
"""测试入口(不依赖 pytest):``python tests/run_tests.py``

带 ``--slow`` 会跑更大规模的随机对照。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_similarity          # noqa: E402
import test_pmfg_exact          # noqa: E402


def run(mod, slow=False):
    names = [n for n in dir(mod) if n.startswith("test_")]
    ok = bad = 0
    for n in sorted(names):
        fn = getattr(mod, n)
        if not callable(fn):
            continue
        if n == "test_random_trials_vs_reference" and not slow:
            trials = 12
        else:
            trials = None
        t0 = time.time()
        try:
            if trials is not None:
                fn(trials=trials)
            else:
                fn()
            print(f"  PASS {mod.__name__}.{n} ({time.time() - t0:.1f}s)")
            ok += 1
        except Exception:
            print(f"  FAIL {mod.__name__}.{n}")
            traceback.print_exc()
            bad += 1
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true", help="跑更大规模的随机对照")
    a = ap.parse_args()
    ok = bad = 0
    for mod in (test_similarity, test_pmfg_exact):
        o, b = run(mod, slow=a.slow)
        ok += o
        bad += b
    print(f"\n== 通过 {ok} 项,失败 {bad} 项")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
