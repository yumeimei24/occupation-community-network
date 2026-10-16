# Method

This document specifies what the package computes, why each step is the way it is,
and where it deviates from the portfolio-herding literature it borrows from.

## 1. From bipartite association data to a node-level similarity graph

Input is a **long-format incidence table** `(node, feature)` — e.g. `(job_posting, task)`,
`(worker, skill)`, `(firm, patent class)`. Duplicates are collapsed, so the incidence
matrix `A` is binary: `A[i, t] = 1` iff node `i` is associated with feature `t`.

Node `i`'s feature set is `T_i = {t : A[i, t] = 1}`, and the similarity between two
nodes is the cosine similarity of their incidence rows,

```
rho_ij = |T_i ∩ T_j| / sqrt(|T_i| |T_j|),
```

which for binary data is the **Ochiai coefficient**. The associated distance is the
one used in the PMFG literature (Tumminello et al., 2005; Aste et al., 2005),

```
d_ij = 2 (1 - rho_ij),
```

which is a monotone decreasing function of `rho_ij`; therefore processing candidate
pairs **in decreasing similarity order** is identical to processing them in increasing
distance order. Only pairs with `rho_ij > 0` (at least one shared feature, hence finite
distance) enter the candidate set. This is the one structural difference from a
complete-graph PMFG: pairs with no shared feature have infinite distance and can never
be added, so they remain isolated singletons (see §5).

## 2. Planar maximally filtered graph (PMFG)

Given the candidate list sorted by decreasing similarity, the greedy PMFG accepts an
edge if and only if the graph remains **planar** after adding it. The result is a
maximal planar subgraph: no further candidate edge can be added without breaking
planarity, and the accepted edges are the strongest associations compatible with
planarity. This is the filtering step that makes the subsequent community detection
non-trivial: the graph keeps only `O(3n)` edges out of up to millions of candidate
pairs.

### 2.1 Exact block-decomposition algorithm

The textbook implementation tests planarity of the **whole graph** for every candidate
edge. With hundreds of thousands to millions of candidates this is the runtime
bottleneck. `tasknet.pmfg.pmfg_block` restricts each test to the part of the graph
that can actually change, using three standard facts about biconnected components
("blocks"):

1. A graph is planar **iff every block is planar** (planarity is a property of the
   disjoint union of blocks).
2. Adding an edge `e = (i, j)` merges exactly the blocks lying on the `i`–`j` path in
   the **block–cut tree** into one new block `B* = (∪ blocks on the path) ∪ {e}`; all
   other blocks are untouched. If `i` and `j` lie in different connected components,
   `e` is a new bridge block and planarity is automatic.
3. Therefore `G + e` is planar **iff for every block `B_m` on that path, the graph
   `B_m` plus the chord `(p_m, q_m)` is planar**, where `p_m, q_m` are the entry/exit
   cut vertices of `B_m` on the path (for the two end blocks, `i` and `j` are used).
   If the chord already is an edge of `B_m`, or `B_m` is a bridge (two vertices, one
   edge), that block passes automatically.

Two exact cheap pre-tests are kept:
Euler's bound `E ≤ 3V − 6` on the block, and the `K5 − e` obstruction (if `i` and `j`
have three mutually adjacent common neighbours, then `{i, j, a, b, c}` induces
`K5 − e`, which is non-planar, so the edge is rejected without a planarity call).

Planarity itself is decided by `rustworkx.is_planar` (Brandes' linear-time algorithm).
The implementation maintains one `rustworkx` graph per block, with a local vertex
numbering, and tests a candidate chord by *add → test → roll back*.

**Equivalence.** The algorithm changes only *where* planarity is decided — never the
accept/reject decision for any candidate. `validation/random_trials.py` and
`tests/test_pmfg_exact.py` verify this empirically against the naive whole-graph
implementation on random and community-structured incidence data, including
tie-shuffled candidate orders; see `docs/validation.md`.

### 2.2 What we deliberately did **not** use

An embedding-based shortcut (maintain a rotation system, accept a chord iff its
endpoints share a face) is *sound but not complete*: by Whitney's theorem the face
structure is unique only for **3-connected** graphs, so for 2-connected blocks with
2-vertex cuts the face test can reject edges the greedy accepts. Measured on real data
this changed ~13% of the edge set, i.e. it is not equivalent — hence the block
decomposition above, which is exact.

## 3. Community detection

The PMFG is partitioned with **Louvain** (`igraph.Graph.community_multilevel`,
Blondel et al., 2008), i.e. modularity maximisation with the standard multi-level
heuristic, on the *unweighted* filtered graph (`resolution = 1` by default; the
similarity values are used for filtering, not for the modularity objective). Isolated
nodes form singleton communities and are reported as such rather than dropped.

Because community labels from modularity maximisation can be compared across runs, the
package reports the statistics that matter for interpretation:

* number of communities and their size distribution;
* modularity;
* mean similarity of pairs inside vs. across communities (using the candidate list),
  which is the "is the partition real?" diagnostic;
* the share of nodes/postings inside communities of size ≥ k (herding intensity);
* the average pairwise Jaccard overlap of communities' core-feature sets, i.e. whether
  several communities do the same thing or different things.

## 4. Herding (core) features

Following the portfolio-herding design, features are ranked **within each community**
by their prevalence `k / n_nodes` (share of the community's nodes that have the
feature). The top 10%, 20% and 30% of the within-community ranking (default headline:
**top 20%**) and communities with `n_nodes ≥ 2` define the community's **core
features** — the analogue of "herded stocks" in the fund-holding application, i.e. the
tasks that a job cluster is built around.

## 5. Connectivity is not herding

Filtered graphs are sparse by construction and contain many **bridges** (cut edges):
chain-like, indirect associations. Counting all connected nodes as "herded" would
therefore overstate herding. The package separates the two notions:

* **blocks** (2-connected components) — sets of nodes with ≥ 2 internally
  vertex-disjoint paths to each other; the graph-theoretic counterpart of herding;
* **within-community edge density** — realised edges over possible edges inside each
  community;
* **bridge share** — fraction of filtered edges that are cut edges;
* **concentration** — HHI of the node weight (e.g. postings) across communities.

## 6. Discrete similarities, ties, and what we do about it

Binary similarity takes very few distinct values: with one shared feature,
`rho = 1/sqrt(|T_i||T_j|)`. In a large job–task network we observe a few hundred
distinct values among hundreds of thousands of candidate pairs, with individual ties
containing thousands of pairs. **The PMFG is therefore not uniquely defined**: any
order within a tie group yields an equally admissible maximal planar graph.

`validation/tie_order_robustness.py` measures how much this matters, by re-running the
whole pipeline with uniformly randomised order inside tie groups (synthetic data,
500 nodes, 5 draws):

| similarity | distinct values | tied edges | edge-set Jaccard vs baseline | edges | modularity | NMI of partition |
|---|---|---|---|---|---|---|
| binary Ochiai | few hundred | large | **0.63 – 0.69** | 1,389 – 1,413 | 0.828 – 0.843 | 0.84 – 0.86 |
| IDF-weighted cosine | ≈ #candidates | ≈ 0.1% | **1.000** | 1,360 (constant) | 0.8522 (constant) | **1.000** |

Two conclusions:

1. **With binary similarity, roughly a third of the filtered edge set is an artefact of
   one arbitrary tie break.** Aggregate structure is fairly stable (modularity,
   community count, size distribution move only slightly), but edge-level and
   community-level claims are not identified.
2. **IDF weighting removes the problem.** Down-weighting ubiquitous features by
   `w_t = log((N+1)/(df_t+1)) + 1` makes the similarity nearly continuous, ties vanish
   (≈0.1% of candidate edges), and the pipeline becomes an exact deterministic function
   of the incidence data. It is also the more sensible similarity for this kind of data:
   sharing a rare, specific task should count for more than sharing a generic one.

`tasknet` therefore defaults to `weighting="idf"` and keeps `weighting="binary"` for
replication of the tie-prone variant, with `validation/tie_order_robustness.py` as the
documented sensitivity check. Louvain is called with a fixed RNG seed, because
`igraph.community_multilevel` is itself randomised: without a seed the same graph gives
different partitions across calls (verified in `tests/`).

## 7. References

* Tumminello, M., Aste, T., Di Matteo, T., Mantegna, R. N. (2005). A tool for
  filtering information in complex systems. *PNAS* 102(30), 10421–10426.
* Aste, T., Di Matteo, T., Tumminello, M., Mantegna, R. N. (2005). Correlation
  filtering and taxonomies in financial networks.
* Blondel, V. D., Guillaume, J.-L., Lambiotte, R., Lefebvre, E. (2008). Fast
  unfolding of communities in large networks. *J. Stat. Mech.*
* Brandes, U. (2007). The left-right planarity test.
* Whitney, H. (1932). Non-separable and planar graphs. *Trans. AMS* 34, 339–362.
* 田正磊, 刘洋溢, 罗荣华 (2025). 基金抱团的成因、演化与经济后果研究——基于网络社区的方法.
  《管理世界》2025 年第 6 期. (The portfolio-herding design this package ports.)
