# -*- coding: utf-8 -*-
"""相似度与候选边构造的单元测试。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from tasknet import ochiai_candidates                      # noqa: E402
from tasknet.similarity import incidence_from_long         # noqa: E402


def test_ochiai_values():
    # 节点0={0,1,2}, 节点1={1,2}, 节点2={2,3}
    rows = [0, 0, 0, 1, 1, 2, 2]
    cols = [0, 1, 2, 1, 2, 2, 3]
    A = incidence_from_long(np.array(rows), np.array(cols), n_jobs=3, n_tasks=4)
    ii, jj, ss = ochiai_candidates(A)
    got = {(int(a), int(b)): float(s) for a, b, s in zip(ii, jj, ss)}
    exp = {(0, 1): 2 / np.sqrt(3 * 2), (0, 2): 1 / np.sqrt(3 * 2), (1, 2): 1 / np.sqrt(2 * 2)}
    assert set(got) == set(exp), (set(got), set(exp))
    for k, v in exp.items():
        assert abs(got[k] - v) < 1e-6, (k, got[k], v)


def test_sorted_desc_and_unique_pairs():
    rng = np.random.default_rng(1)
    rows = np.repeat(np.arange(60), 5)
    cols = rng.integers(0, 40, size=len(rows))
    A = incidence_from_long(rows, cols, n_jobs=60, n_tasks=40)
    ii, jj, ss = ochiai_candidates(A)
    assert np.all(np.diff(ss) <= 1e-6), "候选边必须按相似度降序"
    pairs = list(zip(ii.tolist(), jj.tolist()))
    assert len(pairs) == len(set(pairs)), "同一对只应出现一次"
    assert all(a < b for a, b in pairs), "统一取 i<j"


def test_tie_order_deterministic():
    rng = np.random.default_rng(7)
    rows = np.repeat(np.arange(40), 6)
    cols = rng.integers(0, 12, size=len(rows))       # 特征少 → 大量并列
    A = incidence_from_long(rows, cols, n_jobs=40, n_tasks=12)
    a = ochiai_candidates(A)
    b = ochiai_candidates(A)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.allclose(a[2], b[2])
    vals, counts = np.unique(np.round(a[2], 8), return_counts=True)
    assert counts.max() > 10, "该构造应产生明显并列,用于检验顺序稳定性"
