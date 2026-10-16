# tasknet — PMFG filtering + Louvain communities for job–task (bipartite) networks

`tasknet` turns bipartite association data — most importantly **job postings × tasks**,
but equally workers × skills or firms × technologies — into a filtered similarity graph
and a **community ("herding") structure**, and then characterises which features each
community is built around.

It ports the network-community design used in the portfolio-herding literature
(correlation network → PMFG filtering → Louvain communities → top-quantile "herded"
holdings; see 田正磊, 刘洋溢, 罗荣华, 2025, *管理世界*) to **binary association**
networks, and adds what that setting needs: an exact fast PMFG for millions of
candidates, explicit treatment of similarity ties, and a separation of *connectivity*
from *herding*.

> This repository accompanies a submission under double-blind review. Author names,
> affiliations and acknowledgements are intentionally omitted.

---

## Features

| | |
|---|---|
| **Exact fast PMFG** | Block-decomposition algorithm: each candidate edge is tested only on the biconnected blocks that can change, instead of on the whole graph. Identical edge set to the naive whole-graph greedy (verified on 204 random trials and on a full real-data instance, see `docs/validation.md`). The number of accept/reject decisions is unchanged; the saving is in per-test graph size. |
| **Binary-aware similarity** | Two schemes: IDF-weighted cosine (default — ties negligible, pipeline exactly reproducible) and plain binary Ochiai (tie-prone; kept for replication). Chunked sparse computation of all positive-similarity pairs, deterministic tie ordering. |
| **Tie-order robustness** | `validation/tie_order_robustness.py` re-runs everything with randomised order inside similarity ties and reports edge-set Jaccard, NMI/ARI of the partition and core-feature Jaccard. Measured: binary similarity leaves ~⅓ of the edge set to the tie break; IDF weighting leaves 0%. |
| **Community core features** | Within-community prevalence ranking; top 10/20/30% of features = the community's "herded"/core features. |
| **Connectivity ≠ herding** | Block decomposition, bridges, within-community edge density, concentration (HHI), so that chain-like connectivity is not reported as herding. |
| **External validity tools** | Edge homophily against node attributes (firm, industry, occupation…) with an analytical random baseline, and NMI/ARI/purity of communities against those attributes. |

## Install

```bash
python -m pip install -r requirements.txt      # numpy, scipy, pandas, igraph, rustworkx, networkx
python -m pip install -e .                     # optional: installs the package + `tasknet` CLI
```

Python ≥ 3.9, CPU only. `matplotlib` (plots) and `scikit-learn` (NMI/ARI in scripts) are
listed in `requirements.txt`; the core library does not need them.

## Quickstart

```python
import pandas as pd
from tasknet import prepare_incidence, run_network

# long format: one row per (job posting, task) pair
df = pd.read_csv("incidence.csv")           # columns: job_id, task_id
inc, nodes, feats = prepare_incidence(df, "job_id", "task_id")

meta = pd.read_csv("job_meta.csv").set_index("job_id").loc[nodes].reset_index(drop=True)
res = run_network(inc, node_meta=meta, weight_col="n_postings",
                  attrs_cols=["firm", "industry", "occupation"])

res["stats"]          # communities, modularity, within- vs cross-community similarity
res["core"]           # per community: features ranked by prevalence, core10/20/30 flags
res["blocks"]         # 2-connected cores, bridges, saturation
res["homophily"]      # observed vs random edge homophily
res["ei"], res["ej"], res["ew"]      # filtered (PMFG) edges
res["membership"]                    # community label per node
```

Command line:

```bash
# full pipeline on synthetic data (writes CSV outputs + an overview figure)
python -m tasknet demo --out out/demo

# your own data
python -m tasknet run --incidence incidence.csv --node-col job_id --feature-col task_id \
                      --meta job_meta.csv --weight-col n_postings --out out/run

# correctness evidence: block PMFG vs naive greedy PMFG
python -m tasknet validate --trials 204
```

Outputs: `pmfg_edges.csv`, `communities.csv`, `core_tasks.csv`, `community_stats.csv`,
`edge_summary.csv`, `block_decomposition.csv`, `community_density.csv`,
`edge_homophily.csv`, `overview.png`.

## Method in one page

1. **Similarity.** Node `i` has feature set `T_i`. Default: IDF-weighted cosine,
   `rho_ij = Σ_t w_t² A_it A_jt / (√(Σ_t w_t² A_it) · √(Σ_t w_t² A_jt))` with
   `w_t = log((N+1)/(df_t+1)) + 1`, which is nearly continuous; `weighting="binary"`
   reproduces the plain Ochiai coefficient, whose heavy ties leave the PMFG
   under-identified. Distance `d_ij = 2(1 - rho_ij)`, processed in decreasing similarity
   order (identical to increasing distance).
2. **PMFG.** Accept an edge iff the graph stays planar → a maximal planar graph of the
   strongest associations (`≤ 3n − 6` edges), computed exactly by block decomposition.
3. **Louvain.** Modularity-maximising communities on the filtered graph, with canonical
   edge order and a fixed RNG seed so that results are reproducible; isolated nodes
   remain singletons.
4. **Herding features.** Within each community of size ≥ 2, features ranked by
   prevalence; the top 20% are the community's core ("herded") features.
5. **Diagnostics that keep the interpretation honest.** Within- vs cross-community mean
   similarity; block decomposition and bridge share (connectivity is not herding);
   publication/staffing concentration; edge homophily with analytical random baselines.

Full derivation, the block-decomposition theorem used for the fast exact PMFG, and the
reasons for rejecting an embedding-based shortcut are in **[docs/method.md](docs/method.md)**;
the evidence for exactness and tie-order stability is in
**[docs/validation.md](docs/validation.md)**.

## Repository layout

```
src/tasknet/          library (similarity, pmfg, community, structure, pipeline, cli)
tests/                unit tests + random equivalence trials (no pytest required)
validation/           random-trial equivalence evidence, tie-order robustness
examples/synthetic.py synthetic job–task generator (no real data)
docs/method.md        method, theorem, design choices
docs/validation.md    validation protocol and results
```

## Reproducibility

```bash
python tests/run_tests.py                      # unit tests + 12 randomised equivalence trials
python tests/run_tests.py --slow               # larger randomised equivalence suite
python validation/random_trials.py --trials 204
python validation/tie_order_robustness.py --runs 5
python -m tasknet demo --out out/demo
```

Every number in `docs/validation.md` can be regenerated by these commands on any
machine, without restricted data. The application to real job postings uses proprietary
scraped vacancy data that we cannot redistribute; the repository therefore ships a
**synthetic generator with the same structure** (`examples/synthetic.py`), and the
equivalence/robustness evidence is stated in a data-independent form.

## Data and code availability

Code: MIT (see `LICENSE`). Synthetic data: generated by `examples/synthetic.py`, so it
is fully reproducible from this repository. Derived, de-identified tables for the
empirical application are available from the authors on request, subject to the terms
of the underlying vacancy data.

## 中文摘要

本仓库把「基金抱团」研究的网络社区方法（余弦相似度 → PMFG 过滤 → Louvain 社区 →
社区内前 20% 为「抱团」对象）移植到**岗位—任务**等二值关联网络，并补足该方法在二值
数据下必需的三个环节：(1) **精确快速 PMFG**——用双连通块分解把平面性检验限制在会
变化的块上，边集与朴素整图贪心完全一致（204 个随机试验 + 一个真实数据实例逐边一致，
见 `docs/validation.md`）;(2) **并列处理**——二值相似度取值极少、并列动辄上千条，
实测并列组内换序会改变约 1/3 的边集（边集 Jaccard 0.63–0.69），即 PMFG 在二值口径下
**不可识别**；仓库默认改用 **IDF 加权余弦**，并列降到 0.1%、边集与划分在换序下
100% 不变，并同时保留二值口径供复现；(3) **区分「连通」与「抱团」**——过滤图中大量
边是桥（链式间接相关），真正的抱团对应 2-连通块与社区内部边密度。方法细节见
`docs/method.md`。

## License

MIT. See `LICENSE` and `NOTICE.md`.
