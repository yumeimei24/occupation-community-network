# -*- coding: utf-8 -*-
"""社区发现(抱团识别)与抱团任务。

对应抱团分析的第二步:在 PMFG 上做 Louvain 社区发现,社区即「抱团群体」;
再看社区内部的属性分布。对岗位—任务网络,社区内 prevalence 排名靠前的任务
即被该群体共同「抱团」的任务。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["louvain", "community_stats", "core_tasks"]


def louvain(n, ei, ej, ew=None, resolution=1.0, canonical=True, seed=42):
    """Louvain 社区发现(igraph 的 multilevel 实现)。

    返回 (membership, modularity)。membership 长度为 n,孤立节点自成社区。

    ``canonical=True`` 时先把边按 ``(min, max)`` 规范排序再建图;``seed`` 用于固定
    igraph 的随机数发生器。二者一起保证:给定同一张图,社区划分完全可复现
    (``community_multilevel`` 本身是随机算法,同一输入重复调用会给出不同划分)。
    ``seed=None`` 则不干预 igraph 的随机数发生器。
    """
    import random

    import igraph as ig
    if seed is not None:
        ig.set_random_number_generator(random.Random(int(seed)))
    ei = np.asarray(ei)
    ej = np.asarray(ej)
    ew = None if ew is None else np.asarray(ew, dtype=float)
    if canonical and len(ei):
        a = np.minimum(ei, ej)
        b = np.maximum(ei, ej)
        order = np.lexsort((b, a))
        ei, ej = a[order], b[order]
        if ew is not None:
            ew = ew[order]
    g = ig.Graph(n=int(n), edges=list(zip(ei.tolist(), ej.tolist())), directed=False)
    if ew is not None and len(ew):
        g.es["weight"] = ew.tolist()
        cl = g.community_multilevel(weights="weight", resolution=resolution)
    else:
        cl = g.community_multilevel(resolution=resolution)
    return np.asarray(cl.membership), float(cl.modularity)


def community_stats(job, mem, ii, jj, ss, weight_col="n_postings"):
    """社区层面的结构统计。

    参数
    ----
    job : DataFrame,行 = 节点,至少含 ``weight_col``
    mem : 社区标签数组
    ii, jj, ss : 候选边与其相似度(用于计算社区内/间平均相似度)

    返回 (stats dict, sizes, exposure_w)
    """
    n = len(job)
    mem = np.asarray(mem)
    sizes = pd.Series(mem).value_counts()
    all_pairs = n * (n - 1) / 2
    within_pairs = float(sum(k * (k - 1) / 2 for k in sizes.values if k >= 2))
    cross_pairs = all_pairs - within_pairs

    ii = np.asarray(ii)
    jj = np.asarray(jj)
    ss = np.asarray(ss, dtype=float)
    if len(ii):
        same = mem[ii] == mem[jj]
        within_sum, cross_sum = float(ss[same].sum()), float(ss[~same].sum())
    else:
        within_sum = cross_sum = 0.0

    w = job[weight_col].to_numpy(dtype=float) if weight_col in job else np.ones(n)
    big = sizes[sizes >= 5].index.to_numpy()
    stats = {
        "n_nodes": n,
        "n_communities": int(sizes.size),
        "n_comm_ge2": int((sizes >= 2).sum()),
        "n_comm_ge5": int((sizes >= 5).sum()),
        "n_comm_ge10": int((sizes >= 10).sum()),
        "n_comm_ge30": int((sizes >= 30).sum()),
        "largest_comm": int(sizes.max()) if len(sizes) else 0,
        "largest_comm_share": float(sizes.max() / n) if n else np.nan,
        "mean_within_sim": within_sum / within_pairs if within_pairs > 0 else np.nan,
        "mean_cross_sim": cross_sum / cross_pairs if cross_pairs > 0 else np.nan,
        "share_weight_comm_ge5": float(w[np.isin(mem, big)].sum() / w.sum())
        if w.sum() > 0 else np.nan,
    }
    stats["within_over_cross"] = (stats["mean_within_sim"] / stats["mean_cross_sim"]
                                  if stats["mean_cross_sim"] else np.nan)
    return stats, sizes


def core_tasks(inc, mem, thresholds=(0.10, 0.20, 0.30), min_comm_jobs=2):
    """社区核心任务(抱团任务)。

    参数
    ----
    inc : DataFrame,列 ``node``(节点下标/编号)与 ``feature``(任务/特征标识)
    mem : 社区标签数组(按节点下标排列)
    thresholds : 社区内 prevalence 排名的分位阈值;``core20`` 为常用定义
    min_comm_jobs : 社区规模下限(过小社区的「核心」不具解释力)

    返回
    ----
    DataFrame:``comm, feature, k, n_nodes, prevalence, rank_in_comm, share_rank, core10/20/30``
    """
    inc = inc.copy()
    mem = np.asarray(mem)
    inc["comm"] = mem[inc["node"].to_numpy().astype(int)]
    cnt = inc.groupby(["comm", "feature"]).size().rename("k").reset_index()
    sizes = pd.Series(mem).value_counts().rename("n_nodes").rename_axis("comm").reset_index()
    cnt = cnt.merge(sizes, on="comm")
    cnt["prevalence"] = cnt["k"] / cnt["n_nodes"]
    cnt = cnt.sort_values(["comm", "prevalence", "k"], ascending=[True, False, False])
    cnt["rank_in_comm"] = cnt.groupby("comm").cumcount() + 1
    cnt["n_features_comm"] = cnt.groupby("comm")["feature"].transform("size")
    cnt["share_rank"] = cnt["rank_in_comm"] / cnt["n_features_comm"]
    for th in thresholds:
        cnt[f"core{int(round(th * 100))}"] = (cnt["share_rank"] <= th) & \
            (cnt["n_nodes"] >= min_comm_jobs)
    cnt["is_core"] = cnt[f"core{int(round(thresholds[len(thresholds) // 2] * 100))}"]
    return cnt
