# -*- coding: utf-8 -*-
"""合成「岗位—任务」数据生成器:用于演示与自检(不含任何真实数据)。

生成机制:若干职能群落,每个群落有自己的任务池;岗位从本群落任务池、
全局通用任务池以及少量噪声任务中抽取任务集合。因此网络天然具有
「群落内相似度高于群落间」的结构,可用来检验方法能否把它识别出来。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_synthetic"]

FUNCS = ["研发设计", "质量管理", "设备维护", "生产调度", "行政人事", "供应链"]
INDUSTRIES = ["非金属矿物制品业", "化学原料和化学制品制造业", "有色金属冶炼和压延加工业",
              "电气机械和器材制造业", "专用设备制造业"]
FIRMS = [f"企业{f:02d}" for f in range(1, 41)]


def make_synthetic(n_jobs=600, n_funcs=6, pool_per_func=45, general_pool=40,
                   noise_tasks=250, tasks_per_job=(4, 12), seed=0):
    """返回 (inc_long, node_meta, task_text)。

    inc_long : DataFrame(job_id, task_id)
    node_meta: DataFrame(job_id, firm, industry, func, n_postings, exposure)
    task_text: DataFrame(task_id, task_text)
    """
    rng = np.random.default_rng(seed)
    funcs = FUNCS[:n_funcs]
    n_tasks = n_funcs * pool_per_func + general_pool + noise_tasks
    task_id = np.arange(n_tasks)
    func_of_task = np.full(n_tasks, -1, dtype=int)
    for k in range(n_funcs):
        func_of_task[k * pool_per_func:(k + 1) * pool_per_func] = k
    general_start = n_funcs * pool_per_func
    noise_start = general_start + general_pool

    rows_job, rows_task = [], []
    job_meta = []
    lo, hi = tasks_per_job
    for j in range(n_jobs):
        k = int(rng.integers(n_funcs))
        m = int(rng.integers(lo, hi + 1))
        n_own = max(1, int(m * 0.70))
        n_gen = max(1, int(m * 0.20))
        n_noise = max(0, m - n_own - n_gen)
        own = rng.choice(pool_per_func, size=min(n_own, pool_per_func), replace=False) \
            + k * pool_per_func
        gen = rng.choice(general_pool, size=min(n_gen, general_pool), replace=False) \
            + general_start
        noi = rng.choice(noise_tasks, size=min(n_noise, noise_tasks), replace=False) \
            + noise_start if n_noise else np.zeros(0, dtype=int)
        ts = np.unique(np.concatenate([own, gen, noi]))
        rows_job.append(np.full(len(ts), j))
        rows_task.append(ts)
        job_meta.append(dict(job_id=f"J{j:05d}", func=funcs[k],
                             industry=INDUSTRIES[int(rng.integers(len(INDUSTRIES)))],
                             firm=FIRMS[int(rng.integers(len(FIRMS)))],
                             n_postings=int(rng.integers(1, 25)),
                             exposure=float(np.clip(rng.normal(0.7, 0.15), 0, 1))))
    inc_long = pd.DataFrame({"job_id": np.concatenate(rows_job),
                             "task_id": np.concatenate(rows_task)})
    inc_long["job_id"] = inc_long["job_id"].map(lambda j: f"J{int(j):05d}")
    node_meta = pd.DataFrame(job_meta)
    tt = []
    for t in task_id:
        k = func_of_task[t]
        if k >= 0:
            tt.append(f"[{FUNCS[k]}] 任务 {t}")
        elif t < noise_start:
            tt.append(f"[通用] 任务 {t}")
        else:
            tt.append(f"[噪声] 任务 {t}")
    task_text = pd.DataFrame({"task_id": task_id, "task_text": tt})
    return inc_long, node_meta, task_text


if __name__ == "__main__":
    inc, meta, tt = make_synthetic()
    print(inc.head())
    print(meta.head())
    print("岗位", meta.shape[0], "任务", tt.shape[0], "关联", inc.shape[0])
