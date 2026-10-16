# -*- coding: utf-8 -*-
"""相似度与候选边构造。

对二值关联网络(岗位—任务、工人—技能、企业—技术……),节点 i 的关联集合记为
$T_i$,两个节点的余弦相似度为

.. math::
    \\rho_{ij} = \\frac{|T_i \\cap T_j|}{\\sqrt{|T_i|\\,|T_j|}}

当关联矩阵为 0/1 时,该式等于 Ochiai 系数。距离定义为
$d_{ij} = 2(1-\\rho_{ij})$,是相似度的单调递减函数,故**按相似度降序**
处理候选边即等价于按距离升序。

注意:二值网络的相似度取值高度离散(例如两集合交集为 1 时
$\\rho = 1/\\sqrt{|T_i||T_j|}$),会出现大量并列(tie)。本模块固定
「相似度降序 + 节点编号升序」的稳定顺序,保证结果可复现;
并列顺序的敏感性由 ``validation/`` 中的换序检验刻画。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["incidence_from_long", "ochiai_candidates", "ochiai_matrix", "idf_weights"]


def idf_weights(A, kind="idf", smooth=True):
    """特征权重,用于把二值相似度变成近似连续的加权余弦。

    二值 Ochiai 相似度取值极少(集合交集为 1 时 $\\rho = 1/\\sqrt{|T_i||T_j|}$),
    会出现成千上万条并列边,PMFG 因此不唯一。给特征加 IDF 权重后,

    .. math::
        \\rho_{ij} = \\frac{\\sum_t w_t^2 A_{it} A_{jt}}
        {\\sqrt{\\sum_t w_t^2 A_{it}}\\sqrt{\\sum_t w_t^2 A_{jt}}}

    取值接近连续,并列基本消失,PMFG 重新成为良定义的对象;同时它把
    「人人都有的通用任务」压低权重,这是本类数据里更合理的相似度。

    ``kind="idf"``: $w_t = \\log((N+1)/(df_t+1)) + 1$;``kind="tfidf"`` 同 idf
    (本包输入为 0/1,故 tf 恒为 1)。
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]
    df = np.asarray((A > 0).sum(axis=0)).ravel().astype(float)
    if smooth:
        w = np.log((n + 1.0) / (df + 1.0)) + 1.0
    else:
        with np.errstate(divide="ignore"):
            w = np.log(n / np.maximum(df, 1.0))
        w[~np.isfinite(w)] = 0.0
    w[df <= 0] = 0.0
    return w


def incidence_from_long(jobs, tasks, n_jobs=None, n_tasks=None):
    """由长表 (job, task) 构造 0/1 关联矩阵(CSR)。"""
    jobs = np.asarray(jobs)
    tasks = np.asarray(tasks)
    if n_jobs is None:
        n_jobs = int(jobs.max()) + 1
    if n_tasks is None:
        n_tasks = int(tasks.max()) + 1
    A = sp.csr_matrix((np.ones(len(jobs), dtype=np.float32), (jobs, tasks)),
                      shape=(int(n_jobs), int(n_tasks)))
    A.data[:] = 1.0
    return A


def ochiai_matrix(A, chunk=512, weights=None):
    """返回 (ii, jj, ss):所有正相似度节点对及其相似度,按相似度降序、同分按 (i,j) 升序。

    约定 ``ii < jj``(每对只出现一次),``ss`` 为 float32 相似度。

    参数
    ----
    A : scipy.sparse 矩阵 (n_nodes, n_features),0/1 或非负
    chunk : 分块行数,控制峰值内存(相似度矩阵按块计算)
    weights : 可选的列权重(见 :func:`idf_weights`);默认全 1,即二值 Ochiai
    """
    A = sp.csr_matrix(A)
    A.data = np.clip(A.data, 0, None)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        assert len(w) == A.shape[1], "weights 长度必须等于特征数"
        A = (A @ sp.diags(w)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    An = (sp.diags(inv) @ A).tocsr()
    n = A.shape[0]

    ii, jj, ss = [], [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        Sc = (An[s:e] @ An.T).tocsr()
        ind, ptr, dat = Sc.indices, Sc.indptr, Sc.data
        for r in range(e - s):
            gi = s + r
            st, en = ptr[r], ptr[r + 1]
            if en <= st:
                continue
            c, v = ind[st:en], dat[st:en]
            m = c > gi                      # 每对只保留一次
            if m.any():
                n_new = int(m.sum())
                ii.append(np.full(n_new, gi, dtype=np.int32))
                jj.append(c[m].astype(np.int32))
                ss.append(v[m].astype(np.float32))
    if not ii:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty.copy(), np.zeros(0, dtype=np.float32)
    ii = np.concatenate(ii)
    jj = np.concatenate(jj)
    ss = np.concatenate(ss)
    order = np.lexsort((jj, ii, -ss))
    return ii[order], jj[order], ss[order]


def ochiai_candidates(A, chunk=512, min_sim=0.0, weights=None):
    """同 :func:`ochiai_matrix`,但可设置相似度下限以压缩候选集,并可传列权重。"""
    ii, jj, ss = ochiai_matrix(A, chunk=chunk, weights=weights)
    if min_sim > 0 and len(ss):
        keep = ss >= min_sim
        ii, jj, ss = ii[keep], jj[keep], ss[keep]
    return ii, jj, ss
