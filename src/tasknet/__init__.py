# -*- coding: utf-8 -*-
"""tasknet —— 岗位—任务网络的 PMFG 过滤与社区(抱团)发现工具包。

模块
----
similarity : 二值关联网络的 Ochiai/余弦相似度候选边
pmfg       : 极大平面过滤图(PMFG)的精确实现(双连通块分解)与参照实现
community  : Louvain 社区发现、社区统计、抱团任务(核心任务)
structure  : 图的块分解、边同质性、社区密度等结构刻画
pipeline   : 端到端流程
cli        : 命令行入口(python -m tasknet)
"""

__version__ = "0.1.0"

from .similarity import (ochiai_candidates, incidence_from_long,     # noqa: F401
                         idf_weights, ochiai_matrix)
from .pmfg import pmfg_block, pmfg_reference, BlockPMFG             # noqa: F401
from .community import louvain, community_stats, core_tasks         # noqa: F401
from .structure import (block_decomposition, edge_homophily,        # noqa: F401
                        community_density)
from .pipeline import run_network, prepare_incidence                # noqa: F401

__all__ = [
    "ochiai_candidates", "incidence_from_long", "idf_weights", "ochiai_matrix",
    "pmfg_block", "pmfg_reference", "BlockPMFG",
    "louvain", "community_stats", "core_tasks",
    "block_decomposition", "edge_homophily", "community_density",
    "run_network", "prepare_incidence",
]
