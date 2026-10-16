# -*- coding: utf-8 -*-
"""精确性验证:块分解 PMFG 与朴素参照实现逐边比对。

每个试验随机生成一个 0/1 关联矩阵(随机图或带群落结构的图),构造候选边,
分别用两种实现构造 PMFG,比较**边集是否完全相同**。两者应当完全一致:
块分解只改变「在哪里做平面性检验」,不改变接受/拒绝的判定。

运行::

    python validation/random_trials.py                 # 60 个试验(快速自检)
    python validation/random_trials.py --trials 204    # 论文中的完整规模
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from ocnet import ochiai_candidates, pmfg_block, pmfg_reference      # noqa: E402
from ocnet.similarity import incidence_from_long                    # noqa: E402


def random_incidence(rng, n_nodes, n_feats, mode="community"):
    """生成随机关联数据:``random`` 为独立伯努利;``community`` 含群落结构。"""
    rows, cols = [], []
    if mode == "random":
        for i in range(n_nodes):
            k = int(rng.integers(1, max(2, n_feats // 4)))
            for t in rng.choice(n_feats, size=min(k, n_feats), replace=False):
                rows.append(i)
                cols.append(int(t))
    else:
        n_comm = max(2, n_nodes // 25)
        pools = [rng.choice(n_feats, size=max(3, n_feats // n_comm), replace=False)
                 for _ in range(n_comm)]
        general = rng.choice(n_feats, size=max(3, n_feats // 8), replace=False)
        for i in range(n_nodes):
            p = pools[int(rng.integers(n_comm))]
            k1 = int(rng.integers(1, 4))
            k2 = int(rng.integers(1, 3))
            ts = np.concatenate([rng.choice(p, size=min(k1, len(p)), replace=False),
                                 rng.choice(general, size=min(k2, len(general)),
                                            replace=False)])
            for t in np.unique(ts):
                rows.append(i)
                cols.append(int(t))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def one_trial(rng, n_nodes, n_feats, mode, shuffle_ties=False):
    rows, cols = random_incidence(rng, n_nodes, n_feats, mode)
    A = incidence_from_long(rows, cols, n_jobs=n_nodes, n_tasks=n_feats)
    ii, jj, ss = ochiai_candidates(A)
    if len(ii) == 0:
        return None
    if shuffle_ties:      # 同相似度内部换序(检验并列顺序敏感性)
        order = np.argsort(ss, kind="stable")
        ii, jj, ss = ii[order], jj[order], ss[order]
        vals, starts = np.unique(ss, return_index=True)
        starts = np.append(starts, len(ss))
        perm = np.arange(len(ss))
        for a, b in zip(starts[:-1], starts[1:]):
            if b - a > 1:
                perm[a:b] = rng.permutation(np.arange(a, b))
        ii, jj, ss = ii[perm], jj[perm], ss[perm]

    ei1, ej1, _, cnt = pmfg_block(n_nodes, ii.astype(np.int64), jj.astype(np.int64),
                                  ss.astype(np.float64))
    ei2, ej2, _ = pmfg_reference(n_nodes, ii, jj, ss)
    s1 = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(ei1, ej1)}
    s2 = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(ei2, ej2)}
    return dict(n_nodes=n_nodes, n_feats=n_feats, mode=mode,
                shuffle_ties=bool(shuffle_ties), n_candidates=len(ii),
                edges_block=len(s1), edges_reference=len(s2),
                identical=bool(s1 == s2), n_missing=len(s2 - s1), n_extra=len(s1 - s2),
                chunk_tests=cnt.get("chunk", 0), euler_prunes=cnt.get("euler", 0),
                k5_prunes=cnt.get("k5", 0))


def main(trials=60, seed=20261016, out=None):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for t in range(trials):
        n_nodes = int(rng.integers(4, 320))
        n_feats = int(rng.integers(3, 90))
        mode = "community" if t % 3 else "random"
        r = one_trial(rng, n_nodes, n_feats, mode, shuffle_ties=(t % 4 == 0))
        if r is None:
            continue
        r["trial"] = t
        rows.append(r)
        if (t + 1) % 10 == 0:
            print(f"  试验 {t + 1}/{trials} | 累计不一致 {sum(1 for x in rows if not x['identical'])}",
                  flush=True)
    df = pd.DataFrame(rows)
    ok = int(df["identical"].sum())
    print(f"\n== 等价性验证:{ok}/{len(df)} 个试验边集完全一致 | 用时 {time.time() - t0:.1f}s")
    print(f"   最大节点数 {df['n_nodes'].max()} | 最大候选边 {df['n_candidates'].max():,} | "
          f"最大接受边 {df['edges_reference'].max():,}")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print("   结果已写出:", out)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results_random_trials.csv"))
    a = ap.parse_args()
    main(trials=a.trials, seed=a.seed, out=a.out)
