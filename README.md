# tasknet — 岗位—任务(二部图)网络的 PMFG 过滤 + Louvain 社区

`tasknet` 把二部关联数据——最主要的是**岗位 × 任务**,同样适用于工人 × 技能、
企业 × 技术——变成一张过滤后的相似度网络,识别出**社区(抱团群体)**,并刻画每个
社区是围绕哪些特征组织起来的。

二值关联数据有三个必须处理的问题,这个包就是为它们设计的:

1. **候选边规模巨大**(百万级),朴素 PMFG 逐边检验整图平面性跑不动;
2. **相似度大量并列**(二值余弦只有几百个不同取值),PMFG 因此不唯一;
3. **"连通"不等于"抱团"**,过滤图里很多边是桥(链式间接关联),不能把所有连通节点都算成抱团。

> 本仓库随一份双盲评审的投稿提供,故刻意不出现作者姓名、单位与致谢。

---

## 功能一览

| | |
|---|---|
| **精确的快速 PMFG** | 块分解算法:每条候选边只在"会发生变化"的双连通块上做平面性检验,而不是在整图上做。边集与朴素整图贪心**完全一致**(204 个随机试验 + 一个真实数据实例逐边核对,见 `docs/validation.md`)。省的是每次检验的图规模,不是检验次数。 |
| **二值数据的相似度** | 两种口径:IDF 加权余弦(默认——并列几乎消失,全流程逐位可复现)与纯二值 Ochiai(并列严重,仅为复现保留)。分块稀疏计算全部正相似度对,并列顺序固定可复现。 |
| **并列换序稳健性** | `validation/tie_order_robustness.py`:在相似度并列组内部随机换序、整条管线重跑,报告边集 Jaccard、划分的 NMI/ARI、核心任务集合的 Jaccard。实测:二值口径约 1/3 的边随换序改变;IDF 口径 0%。 |
| **社区核心任务** | 社区内部按任务覆盖率排序,取前 10%/20%/30% 作为该社区的"抱团任务"(默认口径 20%)。 |
| **连通 ≠ 抱团** | 块分解、桥占比、社区内部边密度、集中度(HHI),把链式间接关联和真正的抱团分开。 |
| **外部效度工具** | 边同质性(同社区 / 同企业 / 同行业 / 同岗位大类)对比解析随机基线,以及社区划分对这些属性的 NMI/ARI/纯度。 |

## 安装

```bash
python -m pip install -r requirements.txt      # numpy, scipy, pandas, igraph, rustworkx, networkx
python -m pip install -e .                     # 可选:安装 tasknet 包与命令行
```

Python ≥ 3.9,只需 CPU。`matplotlib`(画图)与 `scikit-learn`(脚本里的 NMI/ARI)也在
`requirements.txt` 中;核心库不依赖它们。

## 快速开始

```python
import pandas as pd
from tasknet import prepare_incidence, run_network

# 长表:一行一个 (岗位, 任务) 对
df = pd.read_csv("incidence.csv")           # 列:job_id, task_id
inc, nodes, feats = prepare_incidence(df, "job_id", "task_id")

meta = pd.read_csv("job_meta.csv").set_index("job_id").loc[nodes].reset_index(drop=True)
res = run_network(inc, node_meta=meta, weight_col="n_postings",
                  attrs_cols=["firm", "industry", "occupation"])

res["stats"]          # 社区数、模块度、社区内 vs 社区间平均相似度
res["core"]           # 每个社区按覆盖率排序的任务,core10/20/30 标记
res["blocks"]         # 2-连通核心块、桥边、饱和度
res["homophily"]      # 观测边同质性 vs 随机基线
res["ei"], res["ej"], res["ew"]   # PMFG 边
res["membership"]                 # 每个节点的社区标签
```

命令行:

```bash
# 用合成数据跑完整管线(输出 CSV + 总览图)
python -m tasknet demo --out out/demo

# 用自己的数据
python -m tasknet run --incidence incidence.csv --node-col job_id --feature-col task_id \
                      --meta job_meta.csv --weight-col n_postings --out out/run

# 正确性证据:块分解 PMFG vs 朴素贪心 PMFG
python -m tasknet validate --trials 204
```

输出文件:`pmfg_edges.csv`、`communities.csv`、`core_tasks.csv`、`community_stats.csv`、
`edge_summary.csv`、`block_decomposition.csv`、`community_density.csv`、
`edge_homophily.csv`、`overview.png`。

## 方法一页纸

1. **相似度**。节点 `i` 的特征集合记为 `T_i`。默认用 IDF 加权余弦
   `rho_ij = Σ_t w_t² A_it A_jt / (√(Σ_t w_t² A_it)·√(Σ_t w_t² A_jt))`,`w_t = log((N+1)/(df_t+1)) + 1`;
   `weighting="binary"` 可复现纯二值 Ochiai 系数 `rho_ij = |T_i∩T_j|/√(|T_i||T_j|)`。
   距离 `d_ij = 2(1 - rho_ij)`,按相似度降序处理候选边(等价于按距离升序)。
2. **PMFG 过滤**。加入某条边后图仍平面才接受 → 只保留最强关联的极大平面子图
   (边数 `≤ 3n − 6`)。用双连通块分解精确实现。
3. **Louvain 社区**。在过滤图上做模块度最大化;固定随机种子并把边按 `(min,max)`
   规范排序,使划分成为"边集"的函数,保证可复现;孤立节点自成一元社区并如实报告。
4. **抱团任务**。在规模 `≥ 2` 的社区内部按覆盖率排序,前 20% 即该社区的核心(抱团)任务。
5. **诚实性的诊断**。社区内 vs 社区间平均相似度;块分解与桥占比(连通不等于抱团);
   发文/岗位集中度;边同质性对比解析随机基线。

完整推导、块分解定理所支撑的快速精确算法、以及为什么放弃"嵌入/共面"捷径,都在
**[docs/method.md](docs/method.md)**;正确性与稳健性证据在
**[docs/validation.md](docs/validation.md)**。

## 目录结构

```
src/tasknet/          库(similarity / pmfg / community / structure / pipeline / cli)
tests/                单元测试 + 随机等价性试验(不需要 pytest)
validation/           随机试验证据、并列换序稳健性
examples/synthetic.py 合成岗位—任务数据生成器(不含任何真实数据)
docs/method.md        方法、定理、设计取舍
docs/validation.md    验证协议与结果
```

## 复现

```bash
python tests/run_tests.py                      # 单元测试 + 12 个随机等价性试验
python tests/run_tests.py --slow               # 更大规模的随机等价性套件
python validation/random_trials.py --trials 204
python validation/tie_order_robustness.py --runs 5 --weighting both
python -m tasknet demo --out out/demo
```

`docs/validation.md` 里的每个数字都可以由上述命令在任何机器上重新生成,不需要任何
受限数据。实证应用使用的是爬虫得到的招聘数据,不能随仓库分发;因此仓库附带一个
**结构相同的合成生成器**(`examples/synthetic.py`),并把等价性与稳健性结论写成与
数据无关的形式。

## 数据与代码可得性

代码:MIT(见 `LICENSE`)。合成数据:由 `examples/synthetic.py` 生成,完全可从本仓库
复现。实证应用衍生的去标识化表格可向作者索取,并受底层招聘数据条款约束。
