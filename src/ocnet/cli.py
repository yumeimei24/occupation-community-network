# -*- coding: utf-8 -*-
"""命令行入口。

    python -m ocnet demo    --out out/demo            # 合成数据全流程演示
    python -m ocnet run     --incidence data.csv --out out/run \
                              --node-col job_id --feature-col task_id
    python -m ocnet validate --trials 60              # 精确性验证(块分解 vs 参照)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _write(outdir, res, task_text=None):
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"i": res["ei"], "j": res["ej"], "similarity": res["ew"]}).to_csv(
        os.path.join(outdir, "pmfg_edges.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"node": np.arange(res["n"]), "community": res["membership"]}).to_csv(
        os.path.join(outdir, "communities.csv"), index=False, encoding="utf-8-sig")
    core = res["core"]
    if task_text is not None:
        core = core.merge(task_text, on="feature", how="left")
    core.to_csv(os.path.join(outdir, "core_tasks.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(res["stats"], modularity=res["modularity"])]).to_csv(
        os.path.join(outdir, "community_stats.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([res["edge_summary"]]).to_csv(
        os.path.join(outdir, "edge_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([res["blocks"]]).to_csv(
        os.path.join(outdir, "block_decomposition.csv"), index=False, encoding="utf-8-sig")
    res["density"].to_csv(os.path.join(outdir, "community_density.csv"),
                          index=False, encoding="utf-8-sig")
    if len(res["homophily"]):
        res["homophily"].to_csv(os.path.join(outdir, "edge_homophily.csv"),
                                index=False, encoding="utf-8-sig")
    print(f"已写出结果到 {outdir}")


def _figure(outdir, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        sizes = pd.Series(res["membership"]).value_counts()
        axes[0].hist(sizes.values, bins=30)
        axes[0].set_title("community size distribution")
        axes[0].set_xlabel("nodes per community")
        dens = res["density"]
        axes[1].scatter(dens["n_nodes"], dens["density"], s=12)
        axes[1].set_xscale("log")
        axes[1].set_title("within-community edge density")
        axes[1].set_xlabel("community size (log)")
        core = res["core"]
        top = core.groupby("feature").size().sort_values(ascending=False).head(20)
        axes[2].barh([str(i) for i in top.index][::-1], top.values[::-1])
        axes[2].set_title("top features by #communities where core")
        fig.tight_layout()
        p = os.path.join(outdir, "overview.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("图已输出:", p)
    except Exception as exc:                                  # pragma: no cover
        print("绘图跳过:", exc)


def cmd_demo(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "examples"))
    from ocnet import prepare_incidence, run_network
    from synthetic import make_synthetic
    inc_long, meta, tt = make_synthetic(seed=args.seed, n_jobs=args.n_jobs)
    inc, nodes, feats = prepare_incidence(inc_long, "job_id", "task_id")
    meta = meta.set_index("job_id").loc[nodes].reset_index(drop=True)
    res = run_network(inc, node_meta=meta, weight_col="n_postings",
                      attrs_cols=["firm", "industry", "func"], log=print)
    _write(args.out, res)
    _figure(args.out, res)
    # 与真实群落标签对照
    pred = pd.Series(res["membership"], name="pred")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    truth = meta["func"].to_numpy()
    print(f"与真实群落标签的一致性:NMI={normalized_mutual_info_score(truth, pred):.3f} "
          f"ARI={adjusted_rand_score(truth, pred):.3f}")


def cmd_run(args):
    from ocnet import prepare_incidence, run_network
    df = pd.read_csv(args.incidence)
    inc, nodes, feats = prepare_incidence(df, args.node_col, args.feature_col)
    meta = None
    if args.meta:
        m = pd.read_csv(args.meta)
        m = m.set_index(args.node_col).loc[nodes].reset_index()
        meta = m.drop(columns=[args.node_col])
    res = run_network(inc, node_meta=meta, weight_col=args.weight_col,
                      min_sim=args.min_sim, resolution=args.resolution,
                      method=args.method, weighting=args.weighting, log=print)
    _write(args.out, res)


def cmd_validate(args):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "validation"))
    import random_trials
    random_trials.main(trials=args.trials, seed=args.seed,
                       out=args.out or os.path.join(here, "validation",
                                                    "results_random_trials.csv"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocnet", description="岗位—任务网络社区发现工具包")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成数据全流程演示")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--n-jobs", type=int, default=600)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="在自定义关联数据上运行")
    r.add_argument("--incidence", required=True, help="长表 CSV:一列节点、一列特征")
    r.add_argument("--node-col", default="job_id")
    r.add_argument("--feature-col", default="task_id")
    r.add_argument("--meta", default=None, help="可选的节点属性 CSV")
    r.add_argument("--weight-col", default=None)
    r.add_argument("--min-sim", type=float, default=0.0)
    r.add_argument("--resolution", type=float, default=1.0)
    r.add_argument("--method", default="block", choices=["block", "reference"])
    r.add_argument("--weighting", default="idf", choices=["idf", "binary"])
    r.add_argument("--out", default="out/run")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="块分解 PMFG 与参照实现的等价性验证")
    v.add_argument("--trials", type=int, default=60)
    v.add_argument("--seed", type=int, default=20261016)
    v.add_argument("--out", default=None)
    v.set_defaults(func=cmd_validate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
