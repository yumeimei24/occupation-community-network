# Validation

Everything here is reproducible from the repository (no data required beyond the
synthetic generator).

## 1. Equivalence of the block-decomposition PMFG and the naive greedy PMFG

**Claim.** For the same ordered candidate list, `pmfg_block` and `pmfg_reference`
accept exactly the same edge set. The block decomposition changes *where* planarity is
tested, not *which* edges are accepted.

**Protocol.** For each random trial: draw a random incidence matrix (two regimes —
independent Bernoulli associations, and a community-structured generator with shared
feature pools plus a general pool), build the ordered candidate list with
`ochiai_candidates`, run both implementations, and compare the edge sets as sets of
unordered pairs. Trials vary the number of nodes, the number of features, the number
of candidates, and (in every fourth trial) randomly reshuffle the order *within*
similarity ties, which is the hardest case for the block bookkeeping.

```bash
python validation/random_trials.py --trials 204        # full evidence (~15 min)
python validation/random_trials.py --trials 60         # quick check
python tests/run_tests.py                              # unit tests + 12 random trials
```

**Result.** `validation/results_random_trials.csv` records one row per trial:
`n_nodes, n_feats, mode, shuffle_ties, n_candidates, edges_block, edges_reference,
identical, n_missing, n_extra`. `identical == True` and `n_missing == n_extra == 0`
for all trials. The bundled unit tests additionally check hand-constructed cases:

| case | expected | reason |
|---|---|---|
| `K5` (all 10 pairs as candidates) | 9 edges accepted | `K5 − e` is the maximal planar graph on 5 vertices (triangular bipyramid) |
| `K3,3` | 8 of 9 accepted | non-planar graph; the greedy saturates at 8 |
| star / tree | all accepted | trees are planar |
| bridge + triangle + chords | identical to reference | exercises the block–cut path merge |

## 2. Real-data check (application to a job–task network)

On a real job–task network (6,147 job nodes, 5,041 O*NET task nodes, 243,010 candidate
pairs), the block-decomposition implementation and the naive whole-graph reference
implementation return the **same 14,954 edges — identical sets, not merely identical
counts** — and the final graph passes an independent whole-graph planarity check
(`rustworkx.is_planar`).

What the block version saves is per-test *size*, not the number of tests: every
same-component candidate still needs one accept/reject decision. In this instance the
largest 2-connected block contains 3,979 nodes (65% of the node set) and 11,321 edges
(76% of the edge set), and there are 871 blocks in total, so a whole-graph test is
replaced by a test on the affected path blocks — i.e. at most ~65–76% of the graph, and
much less for the 519 bridge blocks and the many small blocks. Measured wall-clock on
this instance was within ~20–35% between the two implementations on a partly loaded
16-core machine, in favour of the block version in the cleanest comparison (819 s vs
~1,000 s); the advantage grows with the number of small components (as in the synthetic
trials) and is bounded by the size of the giant 2-connected core, which is the real
cost driver at this scale.

*Note on provenance:* this check ran on the authors' job-posting data, which is not
redistributed in this repository. The synthetic generator and the random-trial suite
reproduce the same equivalence property without any restricted data.

## 3. Tie-order robustness

```bash
python validation/tie_order_robustness.py --runs 5 --weighting both
```

Re-runs the pipeline on the same candidate set with uniformly randomised order inside
each similarity-tie group and reports, per run: number of accepted edges, Jaccard
overlap of the edge set with the baseline, modularity, number of communities, NMI/ARI
of the community partition against the baseline, and average pairwise Jaccard of
community core-feature sets. Output: `results_tie_order_robustness.csv`.

Synthetic data (500 nodes, 5 randomised draws) — the two similarity schemes behave
qualitatively differently:

| weighting | edge Jaccard vs baseline | n_edges | modularity | n_communities | NMI vs baseline |
|---|---|---|---|---|---|
| `binary` (Ochiai) | 0.63 – 0.69 | 1,389 – 1,413 | 0.828 – 0.843 | 11 – 12 | 0.84 – 0.86 |
| `idf` (default) | **1.000** | 1,360 (all runs) | 0.8522 (all runs) | 14 (all runs) | **1.000** |

Reading: binary similarity leaves about a third of the filtered graph to the arbitrary
order inside tie groups, so its edge set is not identified (aggregate statistics are
nevertheless stable). IDF weighting makes ties negligible, and with the seeded Louvain
the pipeline reproduces bit-for-bit across draws. The script asserts, for every
shuffled draw, that the similarity multiset is unchanged and that the order is still
non-increasing — the two mistakes that silently turn "shuffle ties" into "reverse the
whole candidate list" and fake a large sensitivity.

The equivalent real-data check on the authors' job–task network is reported in the
paper; the script accepts `--incidence` so it can be re-run on any table.

## 4. Determinism of the community step

`igraph.community_multilevel` is randomised: calling it repeatedly on the *same* graph
returns different partitions (checked in `tests/debug_louvain_order.py`-style probes and
in the development log of this repository). `tasknet.louvain` therefore

* canonicalises the edge order (`canonical=True`, sorting by `(min, max)`), so the
  partition is a function of the edge *set* rather than of the insertion order, and
* fixes the igraph RNG (`seed=42` by default).

With both, repeated runs on the same graph agree exactly (`n_edges`, modularity,
partition).
