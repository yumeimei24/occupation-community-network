# -*- coding: utf-8 -*-
"""网络结构刻画:块分解、社区密度、边的属性同质性。

核心动机:在 PMFG 这类「过滤图」上,**连通 ≠ 抱团**。过滤图为了保证稀疏性会保留
大量桥(割边),桥只表示链式的间接相关;真正对应「抱团」的是 2-连通块
(块内任意两点之间存在多条路径)与社区内部边密度。本模块把两者分开度量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["block_decomposition", "community_density", "edge_homophily",
           "edge_summary"]


def _to_networkx(n, ei, ej):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(int(n)))
    g.add_edges_from(zip(np.asarray(ei).tolist(), np.asarray(ej).tolist()))
    return g


def block_decomposition(n, ei, ej):
    """图的块分解与连通/抱团指标。

    返回字典:
    ``n_blocks, n_blocks_ge5, n_blocks_ge10, core_V, core_E, core_saturation``
    (最大块边数 / (3V-6)),``nodes_in_core_share, nodes_in_blocks_ge5_share,
    n_bridges, bridge_edge_share, n_components, largest_component_share``
    """
    import networkx as nx
    g = _to_networkx(n, ei, ej)
    n = int(n)
    blocks = list(nx.biconnected_components(g))
    sizes = sorted((len(b) for b in blocks), reverse=True)
    core = max(blocks, key=len) if blocks else set()
    core_e = g.subgraph(core).number_of_edges() if core else 0
    bridges = list(nx.bridges(g)) if g.number_of_edges() else []
    comps = list(nx.connected_components(g))
    return {
        "n_blocks": len(blocks),
        "n_blocks_ge5": int(sum(1 for s in sizes if s >= 5)),
        "n_blocks_ge10": int(sum(1 for s in sizes if s >= 10)),
        "core_V": len(core),
        "core_E": int(core_e),
        "core_saturation": (core_e / (3 * len(core) - 6)) if len(core) >= 3 else np.nan,
        "nodes_in_core_share": len(core) / n if n else np.nan,
        "nodes_in_blocks_ge5_share": (sum(s for s in sizes if s >= 5) / n) if n else np.nan,
        "n_bridges": len(bridges),
        "bridge_edge_share": (len(bridges) / g.number_of_edges()
                              if g.number_of_edges() else np.nan),
        "n_components": len(comps),
        "largest_component_share": (max((len(c) for c in comps), default=0) / n)
        if n else np.nan,
    }


def community_density(ei, ej, mem, weight=None):
    """各社区的内部边密度(社区内实际边数 / 该社区最大可能边数)。

    返回 DataFrame:``comm, n_nodes, n_internal_edges, density``
    """
    ei = np.asarray(ei)
    ej = np.asarray(ej)
    mem = np.asarray(mem)
    cu, cv = mem[ei], mem[ej]
    same = cu == cv
    e_in = pd.Series(cu[same]).value_counts() if same.any() else pd.Series(dtype=int)
    sizes = pd.Series(mem).value_counts().sort_index()
    rows = []
    for cid, k in sizes.items():
        poss = k * (k - 1) / 2
        rows.append(dict(comm=int(cid), n_nodes=int(k),
                         n_internal_edges=int(e_in.get(cid, 0)),
                         density=(e_in.get(cid, 0) / poss) if poss > 0 else np.nan))
    return pd.DataFrame(rows)


def edge_homophily(ei, ej, attrs, cols=None, mem=None):
    """边的属性同质性:观测同属性边占比 vs 随机配对基准 $\\sum_k p_k^2$。

    参数
    ----
    attrs : DataFrame,行 = 节点,列 = 分类属性(企业、行业、职能……)
    cols  : 要考察的属性列;默认全部列
    mem   : 可选,社区标签(额外给出「同社区」一行)

    返回 DataFrame:``dim, observed, expected_random, ratio, n_edges``
    """
    attrs = attrs.reset_index(drop=True)
    ei = np.asarray(ei).astype(int)
    ej = np.asarray(ej).astype(int)
    a = attrs.iloc[ei].reset_index(drop=True)
    b = attrs.iloc[ej].reset_index(drop=True)
    cols = list(attrs.columns) if cols is None else list(cols)
    rows = []
    for c in cols:
        va, vb = a[c].to_numpy(), b[c].to_numpy()
        valid = pd.notna(va) & pd.notna(vb)
        if not valid.any():
            continue
        obs = float(np.mean(va[valid] == vb[valid]))
        vc = attrs[c].value_counts()
        p = (vc / vc.sum()).to_numpy()
        exp = float(np.sum(p ** 2))
        rows.append(dict(dim=c, observed=obs, expected_random=exp,
                         ratio=(obs / exp if exp > 0 else np.nan),
                         n_edges=int(valid.sum())))
    if mem is not None:
        mem = np.asarray(mem)
        obs = float(np.mean(mem[ei] == mem[ej]))
        vc = pd.Series(mem).value_counts()
        p = (vc / vc.sum()).to_numpy()
        exp = float(np.sum(p ** 2))
        rows.append(dict(dim="community", observed=obs, expected_random=exp,
                         ratio=(obs / exp if exp > 0 else np.nan), n_edges=len(ei)))
    return pd.DataFrame(rows)


def edge_summary(n, ei, ej):
    """边数与平面容量、填充率、平均度等基础指标。"""
    ei = np.asarray(ei)
    m = len(ei)
    cap = 3 * int(n) - 6
    return dict(n_nodes=int(n), n_edges=int(m), planar_capacity=int(cap),
                fill_rate=(m / cap if cap > 0 else np.nan),
                mean_degree=(2 * m / n if n else np.nan))
