# -*- coding: utf-8 -*-
"""PMFG 的单元测试:手工构造 + 与参照实现的随机对照。"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "validation"))
from ocnet import pmfg_block, pmfg_reference              # noqa: E402
from random_trials import one_trial                         # noqa: E402


def _edgeset(ei, ej):
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(ei, ej)}


def test_k5_saturates():
    """K5 的 PMFG 应为 9 条边(= 3n-6,三角双锥),且与参照实现一致。"""
    n = 5
    ii, jj, ss = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            ii.append(i)
            jj.append(j)
            ss.append(1.0)
    ei, ej, _, _ = pmfg_block(n, ii, jj, ss)
    assert len(ei) == 9, len(ei)
    ei2, ej2, _ = pmfg_reference(n, ii, jj, ss)
    assert _edgeset(ei, ej) == _edgeset(ei2, ej2)


def test_k33_nonplanar():
    """K3,3 非平面:贪心应接受 8 条边(9 条中拒绝 1 条),两实现一致。"""
    n = 6
    left, right = [0, 1, 2], [3, 4, 5]
    ii, jj, ss = [], [], []
    for a in left:
        for b in right:
            ii.append(a)
            jj.append(b)
            ss.append(1.0)
    ei, ej, _, _ = pmfg_block(n, ii, jj, ss)
    assert len(ei) == 8, len(ei)
    ei2, ej2, _ = pmfg_reference(n, ii, jj, ss)
    assert _edgeset(ei, ej) == _edgeset(ei2, ej2)


def test_star_and_tree():
    """星形/树:所有边都平面,应全部接受。"""
    n = 12
    ii, jj, ss = [], [], []
    for j in range(1, n):
        ii.append(0)
        jj.append(j)
        ss.append(1.0 - 0.01 * j)
    ei, ej, _, _ = pmfg_block(n, ii, jj, ss)
    assert len(ei) == n - 1


def test_bridge_plus_cycle():
    """桥接块 + 环:先建三角形(0,1,2),再把 3、4 用桥接边连上,再考察弦。"""
    n = 6
    cand = [(0, 1, 1.0), (1, 2, 0.9), (0, 2, 0.8),   # 三角
            (2, 3, 0.7), (3, 4, 0.6), (4, 5, 0.5),   # 链
            (1, 3, 0.4), (0, 4, 0.3)]                # 弦
    ii = [c[0] for c in cand]
    jj = [c[1] for c in cand]
    ss = [c[2] for c in cand]
    ei, ej, _, _ = pmfg_block(n, ii, jj, ss)
    ei2, ej2, _ = pmfg_reference(n, ii, jj, ss)
    assert _edgeset(ei, ej) == _edgeset(ei2, ej2)


def test_random_trials_vs_reference(trials=24, seed=99):
    """随机对照(含并列换序),块分解必须与参照实现边集完全一致。"""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    n_bad = 0
    for t in range(trials):
        n_nodes = int(rng.integers(4, 160))
        n_feats = int(rng.integers(3, 60))
        r = one_trial(rng, n_nodes, n_feats, "community" if t % 2 else "random",
                      shuffle_ties=(t % 3 == 0))
        if r is None:
            continue
        if not r["identical"]:
            n_bad += 1
            print("   不一致:", r)
    assert n_bad == 0, f"{n_bad} 个试验边集不一致"
    print(f"   {trials} 个随机试验全部一致(用时 {time.time() - t0:.1f}s)")
