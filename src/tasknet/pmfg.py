# -*- coding: utf-8 -*-
"""极大平面过滤图(PMFG)的精确实现。

背景
----
PMFG(Planar Maximally Filtered Graph, Tumminello et al. 2005)按距离(相似度)
顺序逐条考察候选边,仅当加入后图仍保持平面性时才接受,得到的图在「保留平面性」
这一约束下容纳了最强的关联结构。经典算法需要对每个候选边做一次平面性检验;
朴素实现每次都在整张图上检验,当节点数上万、候选边上百万时非常慢。

本模块的 :func:`pmfg_block` 用**双连通块分解**把这步变快,且与朴素贪心
**逐边完全等价**(见 ``docs/method.md`` 与 ``validation/``):

1. 图 G 平面 ⇔ G 的每个双连通块(块)平面;
2. 加入边 e=(i,j) 只把「块—割点树」上 i 到 j 路径上的块合并为一个新块
   B* = (路径诸块之并) ∪ {e},其余块不受影响;
3. 因此 G+e 平面 ⇔ 路径上每个块 B_m 满足「B_m 加入弦 (p_m,q_m) 后仍平面」,
   (p_m,q_m) 是该块的入口/出口割点(两端块分别用 i、j);
4. 若弦 (p_m,q_m) 已是 B_m 的边,或 B_m 是桥块(2 点 1 边),该块自动通过。

由此每次只需在路径上的块(通常远小于整图)上做平面性检验。平面性判定用
``rustworkx.is_planar``(Brandes 线性时间算法)。另保留两个精确的廉价剪枝:
Euler 上界 $E \\le 3V-6$ 与 $K_5-e$ 充分非平面判定。
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rustworkx as rx

__all__ = ["BlockPMFG", "pmfg_block", "pmfg_reference"]


class BlockPMFG:
    """增量维护双连通块结构的精确 PMFG 构造器。

    用法::

        f = BlockPMFG(n)
        for i, j, w in zip(ii, jj, ss):     # 须按相似度降序
            f.add(int(i), int(j), float(w))
        ei, ej, ew = f.finish()
    """

    def __init__(self, n):
        self.n = int(n)
        self.parent = list(range(self.n))            # 并查集:连通分量
        self.blocks = {}                             # bid -> 块字典
        self.v2b = {}                                # 顶点 -> 所属块集合
        self._bid = 0
        self.adj = [set() for _ in range(self.n)]    # 已接受边的邻接表
        self.acc_i, self.acc_j, self.acc_w = [], [], []
        self.edgeset = set()
        self.cnt = dict(chunk=0, euler=0, k5=0, free=0, merge=0)

    # ------------------------------------------------------------------ 并查集
    def find(self, x):
        p = self.parent
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    # -------------------------------------------------------------- 块的基本操作
    def _new_block(self, verts=None, edges=None):
        bid = self._bid
        self._bid += 1
        blk = dict(v=set(), e=set(), loc={}, inv=[], g=rx.PyGraph(multigraph=False))
        self.blocks[bid] = blk
        for v in (verts or ()):
            self._add_vertex(bid, v)
        for e in (edges or ()):
            a, b = tuple(e)
            self._add_edge(bid, a, b)
        return bid

    def _add_vertex(self, bid, v):
        blk = self.blocks[bid]
        if v in blk["loc"]:
            return blk["loc"][v]
        li = blk["g"].add_node(v)
        blk["loc"][v] = li
        blk["inv"].append(v)
        blk["v"].add(v)
        self.v2b.setdefault(v, set()).add(bid)
        return li

    def _add_edge(self, bid, u, v):
        blk = self.blocks[bid]
        self._add_vertex(bid, u)
        self._add_vertex(bid, v)
        blk["g"].add_edge(blk["loc"][u], blk["loc"][v], None)
        blk["e"].add(frozenset((u, v)))

    def _drop_block(self, bid):
        blk = self.blocks.pop(bid)
        for v in blk["v"]:
            s = self.v2b.get(v)
            if s is not None:
                s.discard(bid)
                if not s:
                    self.v2b.pop(v, None)

    def _test_edge_in_block(self, bid, p, q):
        """块 bid 加入弦 (p,q) 后是否仍平面(精确检验,检验后回滚)。"""
        blk = self.blocks[bid]
        if frozenset((p, q)) in blk["e"]:
            return True                                  # 弦已是块内边
        if len(blk["v"]) <= 2:
            return True                                  # 桥块
        self.cnt["chunk"] += 1
        li, lj = blk["loc"][p], blk["loc"][q]
        idx = blk["g"].num_edges()
        blk["g"].add_edge(li, lj, None)
        try:
            ok = rx.is_planar(blk["g"])
        finally:
            blk["g"].remove_edge_from_index(idx)
        return ok

    def _k5e_reject(self, i, j):
        """K5 去掉一条边后仍非平面:若 i、j 有 3 个两两相邻的公共邻居,则可直接拒绝。"""
        ai, aj = self.adj[i], self.adj[j]
        if len(ai) < 3 or len(aj) < 3:
            return False
        small, big = (ai, aj) if len(ai) <= len(aj) else (aj, ai)
        common = [x for x in small if x in big]
        m = len(common)
        if m < 3:
            return False
        for a in range(m - 2):
            na = self.adj[common[a]]
            for b in range(a + 1, m - 1):
                if common[b] not in na:
                    continue
                nb = self.adj[common[b]]
                for c in range(b + 1, m):
                    if common[c] in na and common[c] in nb:
                        return True
        return False

    # ------------------------------------------------------------ 块—割点树路径
    def block_path(self, i, j):
        """返回 (blocks, arts):块—割点树上 i 到 j 的块序列与割点序列。"""
        start = ("v", i)
        prev = {start: None}
        dq = deque([start])
        found = None
        while dq:
            node = dq.popleft()
            if node[0] == "v" and node[1] == j and node != start:
                found = node
                break
            if node[0] == "v":
                for b in self.v2b.get(node[1], ()):
                    nxt = ("b", b)
                    if nxt not in prev:
                        prev[nxt] = node
                        dq.append(nxt)
            else:
                for v in self.blocks[node[1]]["v"]:
                    nxt = ("v", v)
                    if nxt not in prev:
                        prev[nxt] = node
                        dq.append(nxt)
        if found is None:
            return None
        seq = []
        cur = found
        while cur is not None:
            seq.append(cur)
            cur = prev[cur]
        seq.reverse()
        blocks, arts = [], [i]
        for kind, val in seq[1:]:
            if kind == "b":
                blocks.append(val)
            else:
                arts.append(val)
        return blocks, arts

    # ------------------------------------------------------------------ 主判定
    def add(self, i, j, w):
        """考察候选边 (i,j);接受返回 True。"""
        if (i, j) in self.edgeset:
            return False
        ri, rj = self.find(i), self.find(j)
        if ri != rj:                                     # 跨连通分量:一定平面
            self._new_block([i, j], [(i, j)])
            self._record(i, j, w)
            self.parent[rj] = ri
            self.cnt["free"] += 1
            return True
        bi, bj = self.v2b.get(i), self.v2b.get(j)
        common = bi & bj if (bi and bj) else None
        if common:                                       # 同块
            return self._within_block(next(iter(common)), i, j, w)
        return self._across_blocks(i, j, w)              # 不同块但同分量

    def _record(self, i, j, w):
        self.acc_i.append(i)
        self.acc_j.append(j)
        self.acc_w.append(w)
        self.edgeset.add((i, j))
        self.adj[i].add(j)
        self.adj[j].add(i)

    def _within_block(self, bid, i, j, w):
        blk = self.blocks[bid]
        vb, eb = len(blk["v"]), len(blk["e"])
        cap = 3 * vb - 6 if vb >= 3 else (1 if vb == 2 else 0)
        if eb + 1 > cap:
            self.cnt["euler"] += 1
            return False
        if self._k5e_reject(i, j):
            self.cnt["k5"] += 1
            return False
        if not self._test_edge_in_block(bid, i, j):
            return False
        self._add_edge(bid, i, j)
        self._record(i, j, w)
        return True

    def _across_blocks(self, i, j, w):
        path = self.block_path(i, j)
        if path is None:
            return False
        blocks, arts = path
        for m, bid in enumerate(blocks):
            p, q = arts[m], arts[m + 1]
            if p == q:
                continue
            blk = self.blocks[bid]
            if frozenset((p, q)) not in blk["e"]:
                vb, eb = len(blk["v"]), len(blk["e"])
                cap = 3 * vb - 6 if vb >= 3 else (1 if vb == 2 else 0)
                if eb + 1 > cap:
                    self.cnt["euler"] += 1
                    return False
            if not self._test_edge_in_block(bid, p, q):
                return False
        if self._k5e_reject(i, j):
            self.cnt["k5"] += 1
            return False
        self._merge_path(blocks, arts, i, j, w)
        return True

    def _merge_path(self, blocks, arts, i, j, w):
        self.cnt["merge"] += 1
        _, keep = max((len(self.blocks[b]["e"]), b) for b in blocks)
        keep_blk = self.blocks[keep]
        for bid in blocks:
            if bid == keep:
                continue
            src = self.blocks[bid]
            for v in src["v"]:
                self._add_vertex(keep, v)
            for e in src["e"]:
                a, b = tuple(e)
                if frozenset((a, b)) not in keep_blk["e"]:
                    self._add_edge(keep, a, b)
        for bid in blocks:
            if bid != keep:
                self._drop_block(bid)
        self._add_edge(keep, i, j)
        self._record(i, j, w)

    # -------------------------------------------------------------------- 收尾
    def finish(self):
        return (np.array(self.acc_i, dtype=np.int32),
                np.array(self.acc_j, dtype=np.int32),
                np.array(self.acc_w, dtype=np.float32))

    def n_edges(self):
        return len(self.acc_i)


def pmfg_block(n, ii, jj, ss, log=None, max_edges=None):
    """块分解精确 PMFG。

    参数
    ----
    n : 节点数
    ii, jj, ss : 候选边(须按相似度/权重降序传入)
    log : 可调用对象,用于输出统计
    max_edges : 边数上限(默认 3n-6,达到即停;平面图不可能超过该上界)

    返回
    ----
    (ei, ej, ew, counters)
    """
    f = BlockPMFG(n)
    total = len(ii)
    sat = max_edges if max_edges is not None else 3 * n - 6
    for k in range(total):
        if f.n_edges() >= sat:
            if log:
                log(f"    已达 {sat:,} 边上界,提前终止于候选 {k:,}")
            break
        f.add(int(ii[k]), int(jj[k]), float(ss[k]))
    ei, ej, ew = f.finish()
    if log:
        log("  块分解 PMFG:块内平面性检验 {chunk:,} | Euler 剪枝 {euler:,} | "
            "K5-e 剪枝 {k5:,} | 跨分量直连 {free:,} | 块合并 {merge:,}".format(**f.cnt))
    return ei, ej, ew, f.cnt


def pmfg_reference(n, ii, jj, ss, log_every=0, log=None):
    """参照实现:朴素贪心,每个候选边在整张图上做一次平面性检验。

    用于正确性验证(见 ``validation/random_trials.py``),不追求速度。
    """
    g = rx.PyGraph(multigraph=False)
    g.add_nodes_from(range(n))
    parent = list(range(n))

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    ei, ej, ew = [], [], []
    sat = 3 * n - 6
    for k in range(len(ii)):
        if len(ei) >= sat:
            break
        i, j = int(ii[k]), int(jj[k])
        ri, rj = find(i), find(j)
        if ri == rj:
            idx = g.num_edges()
            g.add_edge(i, j, None)
            ok = rx.is_planar(g)
            if not ok:                       # 非平面:回滚并拒绝
                g.remove_edge_from_index(idx)
                continue
            # 平面:保留该边(不回滚)
        else:
            g.add_edge(i, j, None)
            parent[rj] = ri
        ei.append(i)
        ej.append(j)
        ew.append(float(ss[k]))
        if log_every and log and (k + 1) % log_every == 0:
            log(f"    候选 {k + 1:,}/{len(ii):,} | 接受 {len(ei):,}")
    return (np.array(ei, dtype=np.int32), np.array(ej, dtype=np.int32),
            np.array(ew, dtype=np.float32))
