# -*- coding: utf-8 -*-
"""Tie-order robustness of the whole pipeline.

Binary similarity takes very few distinct values, so PMFG is not unique: any order
inside a tie group gives an equally admissible maximal planar graph. This script
quantifies how much the results move when that arbitrary order is randomised.

With no arguments it runs on synthetic data (reproducible); pass ``--incidence`` to run
on your own long-format table. Results are appended to a CSV, so the script is
resumable: interrupt it and start it again.

    python validation/tie_order_robustness.py                      # synthetic, 5 runs
    python validation/tie_order_robustness.py --runs 10 --seed 7
    python validation/tie_order_robustness.py --incidence data.csv \
        --node-col job_id --feature-col task_id --out results_tie_order_robustness.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

from ocnet import (ochiai_candidates, idf_weights, pmfg_block, louvain,    # noqa: E402
                     core_tasks, community_stats, block_decomposition)
from ocnet.similarity import incidence_from_long                          # noqa: E402


def shuffle_within_ties(ii, jj, ss, rng):
    """Keep the similarity order, randomise the order inside each tie group.

    实现要点(易错):``ss`` 是**相似度降序**,不能直接用
    ``np.unique(ss, return_index=True)`` 定位并列段——该函数按升序返回唯一值,
    对降序数组给出的首现下标不是单调递增的,会把不同相似度的边混在一起。
    这里改用相邻等值扫描,并断言换序后仍严格非升。
    """
    ss = np.asarray(ss)
    assert np.all(np.diff(ss) <= 1e-12), "候选边必须按相似度降序传入"
    n = len(ss)
    perm = np.arange(n)
    i = 0
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]:
            j += 1
        if j - i > 1:
            perm[i:j] = rng.permutation(np.arange(i, j))
        i = j
    out = (ii[perm], jj[perm], ss[perm])
    assert np.all(np.diff(out[2]) <= 1e-12), "换序后顺序被破坏"
    assert np.array_equal(np.sort(out[2]), np.sort(ss)), "换序改变了相似度多重集"
    return out


def core_feature_jaccard(core, min_size=3, max_sets=300):
    from itertools import combinations
    sets = [set(g["feature"]) for c, g in core[core["is_core"]].groupby("comm")
            if g["n_nodes"].iloc[0] >= min_size]
    sets = sets[:max_sets]
    if len(sets) < 2:
        return np.nan
    return float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidence", default=None, help="长表 CSV;默认用合成数据")
    ap.add_argument("--node-col", default="job_id")
    ap.add_argument("--feature-col", default="task_id")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=600, help="合成数据的岗位数")
    ap.add_argument("--weighting", default="both",
                    choices=["binary", "idf", "both"],
                    help="binary:二值 Ochiai;idf:IDF 加权余弦;both:两者都跑并对比")
    ap.add_argument("--out", default=os.path.join(HERE, "results_tie_order_robustness.csv"))
    a = ap.parse_args()

    if a.incidence:
        df = pd.read_csv(a.incidence)
        nodes = pd.Index(pd.unique(df[a.node_col]))
        feats = pd.Index(pd.unique(df[a.feature_col]))
        node_ids = nodes.get_indexer(df[a.node_col]).astype(np.int32)
        feat_ids = feats.get_indexer(df[a.feature_col]).astype(np.int32)
    else:
        from synthetic import make_synthetic
        inc_long, meta, _ = make_synthetic(n_jobs=a.n_jobs, seed=a.seed)
        nodes = pd.Index(pd.unique(inc_long["job_id"]))
        feats = pd.Index(pd.unique(inc_long["task_id"]))
        node_ids = nodes.get_indexer(inc_long["job_id"]).astype(np.int32)
        feat_ids = feats.get_indexer(inc_long["task_id"]).astype(np.int32)

    n, n_feat = len(nodes), len(feats)
    A = incidence_from_long(node_ids, feat_ids, n_jobs=n, n_tasks=n_feat)
    inc = pd.DataFrame({"node": node_ids, "feature": feat_ids})
    schemes = ["binary", "idf"] if a.weighting == "both" else [a.weighting]

    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        rows = prev.to_dict("records")
        done = {(r["weighting"], r["run"]) for _, r in prev.iterrows()}
        print(f"已有 {len(done)} 次运行,继续追加")

    for scheme in schemes:
        w = idf_weights(A) if scheme == "idf" else None
        ii0, jj0, ss0 = ochiai_candidates(A, weights=w)
        uniq = len(np.unique(np.round(ss0, 8)))
        cnt = np.unique(np.round(ss0, 8), return_counts=True)[1]
        tied = int(cnt[cnt > 1].sum())
        print(f"\n=== 加权口径 {scheme}:候选 {len(ii0):,} | 不同取值 {uniq:,} | "
              f"并列边 {tied:,}({tied / max(len(ss0), 1):.1%})", flush=True)

        base_mem = base_edges = None
        for run in range(a.runs + 1):                   # run 0 = baseline order
            label = "baseline" if run == 0 else f"shuffle{run}"
            if (scheme, label) in done:
                continue
            rng = np.random.default_rng(a.seed * 1000 + run)
            ii, jj, ss = (ii0, jj0, ss0) if run == 0 else \
                shuffle_within_ties(ii0, jj0, ss0, rng)
            t0 = time.time()
            ei, ej, ew, _c = pmfg_block(n, ii.astype(np.int64), jj.astype(np.int64),
                                        ss.astype(np.float64))
            mem, mod = louvain(n, ei, ej, ew)
            core = core_tasks(inc, mem)
            blocks = block_decomposition(n, ei, ej)
            row = dict(weighting=scheme, run=label, n_edges=len(ei),
                       modularity=round(mod, 4),
                       n_communities=int(len(np.unique(mem))),
                       largest_comm=int(pd.Series(mem).value_counts().max()),
                       core_feature_jaccard=round(core_feature_jaccard(core), 4),
                       core_features=int(core["is_core"].sum()),
                       bridge_share=round(blocks["bridge_edge_share"], 4),
                       seconds=round(time.time() - t0, 1))
            if run == 0:
                base_mem = mem
                base_edges = {(min(int(x), int(y)), max(int(x), int(y)))
                              for x, y in zip(ei, ej)}
                row.update(edge_jaccard_vs_base=1.0, nmi_vs_base=1.0, ari_vs_base=1.0)
            else:
                from sklearn.metrics import (normalized_mutual_info_score as nmi,
                                             adjusted_rand_score as ari)
                e_now = {(min(int(x), int(y)), max(int(x), int(y)))
                         for x, y in zip(ei, ej)}
                row["edge_jaccard_vs_base"] = round(len(base_edges & e_now) /
                                                    len(base_edges | e_now), 4)
                row["nmi_vs_base"] = round(float(nmi(base_mem, mem)), 4)
                row["ari_vs_base"] = round(float(ari(base_mem, mem)), 4)
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    print("\n== 并列顺序稳健性汇总 ==")
    print(df.to_string(index=False))
    print("\n结果:", a.out)


if __name__ == "__main__":
    main()
