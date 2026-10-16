# -*- coding: utf-8 -*-
"""端到端流程:相似度 → PMFG 过滤 → Louvain 社区 → 抱团任务与结构刻画。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .similarity import incidence_from_long, ochiai_candidates, idf_weights
from .pmfg import pmfg_block, pmfg_reference
from .community import louvain, community_stats, core_tasks
from .structure import (block_decomposition, community_density, edge_homophily,
                        edge_summary)

__all__ = ["prepare_incidence", "run_network"]


def prepare_incidence(df, node_col="node", feature_col="feature"):
    """把长表整理为整数下标的关联表。

    返回 (inc, nodes, features):``inc`` 含 ``node, feature`` 两列(整数下标),
    ``nodes`` / ``features`` 为下标到原始标识的映射(list)。
    """
    d = df[[node_col, feature_col]].dropna().drop_duplicates()
    nodes = pd.Index(pd.unique(d[node_col]))
    feats = pd.Index(pd.unique(d[feature_col]))
    inc = pd.DataFrame({
        "node": nodes.get_indexer(d[node_col]).astype(np.int32),
        "feature": feats.get_indexer(d[feature_col]).astype(np.int32),
    })
    return inc, list(nodes), list(feats)


def run_network(inc, node_meta=None, weight_col=None, min_sim=0.0,
                resolution=1.0, method="block", thresholds=(0.10, 0.20, 0.30),
                min_comm_jobs=2, attrs_cols=None, weighting="idf", seed=42, log=print):
    """完整流程。

    参数
    ----
    inc : DataFrame,列 ``node``(0..n-1 整数)与 ``feature``
    node_meta : 可选,行 = 节点的 DataFrame(可含权重列与分类属性列,用于同质性分析)
    weight_col : node_meta 中的权重列(如招聘量),用于加权统计;None 则等权
    min_sim : 候选边的相似度下限(默认 0,即保留所有正相似度对)
    method : ``"block"``(精确快速)或 ``"reference"``(朴素参照实现)
    weighting : ``"idf"``(默认)用 IDF 加权余弦;``"binary"`` 用二值 Ochiai
    seed : Louvain 随机种子(保证可复现);None 表示不固定
    attrs_cols : 参与边同质性分析的属性列;默认 node_meta 的全部列

    返回
    ----
    dict,键包括 ``ii, jj, ss, ei, ej, ew, membership, modularity, stats,
    core, blocks, edge_summary, density, homophily``
    """
    n = int(inc["node"].max()) + 1
    n_feat = int(inc["feature"].max()) + 1
    A = incidence_from_long(inc["node"].to_numpy(), inc["feature"].to_numpy(),
                            n_jobs=n, n_tasks=n_feat)
    # weighting="idf": 用 IDF 加权余弦,取值近似连续、并列基本消失,PMFG 良定义;
    # weighting="binary": 二值 Ochiai(原口径),并列极多,PMFG 不唯一(见 docs/method.md §6)
    w = idf_weights(A) if weighting in ("idf", "tfidf") else None
    ii, jj, ss = ochiai_candidates(A, min_sim=min_sim, weights=w)
    if log:
        n_ties = 0
        if len(ss):
            _, cnt = np.unique(np.round(ss, 8), return_counts=True)
            n_ties = int(cnt[cnt > 1].sum())
        log(f"[1/5] 节点 {n:,} | 特征 {n_feat:,} | 关联 {len(inc):,} | 候选边 {len(ii):,} | "
            f"加权={weighting} | 并列边 {n_ties:,}")

    if method == "reference":
        ei, ej, ew = pmfg_reference(n, ii, jj, ss, log=log)
        counters = {}
    else:
        ei, ej, ew, counters = pmfg_block(n, ii, jj, ss, log=log)
    if log:
        log(f"[2/5] PMFG:接受 {len(ei):,} 条边(候选 {len(ii):,})")

    mem, modularity = louvain(n, ei, ej, ew, resolution=resolution, seed=seed)
    if log:
        log(f"[3/5] Louvain:社区 {len(np.unique(mem)):,} | 模块度 {modularity:.4f}")

    meta = node_meta.reset_index(drop=True) if node_meta is not None else \
        pd.DataFrame(index=range(n))
    if weight_col and weight_col not in meta:
        weight_col = None
    stats, sizes = community_stats(meta if len(meta) else pd.DataFrame({"w": np.ones(n)}),
                                   mem, ii, jj, ss,
                                   weight_col=weight_col or "n_postings")
    core = core_tasks(inc, mem, thresholds=thresholds, min_comm_jobs=min_comm_jobs)
    if log:
        log(f"[4/5] 抱团任务:{int(core['is_core'].sum()):,} 条核心任务记录")

    blocks = block_decomposition(n, ei, ej)
    esum = edge_summary(n, ei, ej)
    dens = community_density(ei, ej, mem)
    homo = edge_homophily(ei, ej, meta, cols=attrs_cols, mem=mem) \
        if (node_meta is not None and len(meta.columns)) else pd.DataFrame()
    if log:
        log(f"[5/5] 结构:块 {blocks['n_blocks']:,} | 核心块 {blocks['core_V']:,} 节点 | "
            f"桥边占比 {blocks['bridge_edge_share']:.4f}")

    return dict(n=n, n_features=n_feat, ii=ii, jj=jj, ss=ss, weighting=weighting,
                ei=ei, ej=ej, ew=ew, membership=mem, modularity=modularity,
                stats=stats, sizes=sizes, core=core, blocks=blocks,
                edge_summary=esum, density=dens, homophily=homo,
                counters=counters)
