# ocnet — 岗位群落网络

这个包做一件事:给定"哪些岗位需要哪些任务"的对应关系,把任务清单相近的岗位连成一张网络,在网络上把岗位分成群落,并找出每个群落共同承担的任务。

数据的来源是一批招聘记录。每条记录写明一个岗位需要完成的任务,把同一岗位在记录里出现的任务合并起来,就得到一张"岗位—任务"对应表:一行一个岗位,一列一个任务,格子里的 1 表示该岗位需要该任务。这张表是后面所有计算的唯一输入,不需要任何行业、企业或职业标签。

计算分四步。

1. **算相似度**。两个岗位的任务清单重叠越多、重叠的部分越不常见,就越相似。这样得到的相似度是连续的数值,方便排序和比较。
2. **过滤网络**。把每两个岗位之间的相似度都算出来之后,只保留其中最强的联系:按相似度从高到低逐条尝试加入,加入后图仍然保持平面才留下。最后得到一张只含最强联系的骨干网络,边数大约是岗位数的三倍。
3. **划分群落**。在骨干网络上做社区划分,把联系紧密的岗位归为一群,联系稀疏的各自独立。
4. **找核心任务**。在每一个不少于两个岗位的群落里,统计每个任务被群落内多大比例的岗位需要,比例最高的前 20% 记为该群落的核心任务。核心任务就是这个群落的岗位共同在做的活。

过滤网络这一步的计算量主要来自平面性检验。朴素的做法每加一条边都对整张图检验一次,岗位数量大时很慢;本包改为只在受影响的块上检验,得到的结果与朴素做法完全一致,见 `docs/method.md` 的说明和 `docs/validation.md` 的核对结果。

另外,任务清单是 0/1 数据,相似度取值高度重复,同一批相似度里谁先谁后会影响结果。本包默认用加权的相似度让取值分散开,并固定了并列时的顺序和社区划分的随机种子,使同样的数据每次得到同样的结果。

> 本仓库随一项在审研究提供,不含作者信息。

## 安装

```bash
python -m pip install -r requirements.txt      # numpy, scipy, pandas, igraph, rustworkx, networkx
python -m pip install -e .                     # 可选:安装 ocnet 包与命令行
```

Python 3.9 以上,只需 CPU。

## 用自己的数据

数据准备成一张长表 CSV,两列:岗位编号、任务编号,一行一对。可选再准备一张岗位属性表(企业、行业、岗位类别等),用于核对结果。

```bash
python -m ocnet run --incidence 对应表.csv --node-col 岗位编号 --feature-col 任务编号 \
                    --meta 岗位属性.csv --weight-col 发文量 --out out/run
```

或者在 Python 里:

```python
import pandas as pd
from ocnet import prepare_incidence, run_network

pairs = pd.read_csv("pairs.csv")                  # 列:job_id, task_id
inc, nodes, feats = prepare_incidence(pairs, "job_id", "task_id")

meta = pd.read_csv("jobs.csv").set_index("job_id").loc[nodes].reset_index(drop=True)
res = run_network(inc, node_meta=meta, weight_col="n_postings",
                  attrs_cols=["firm", "industry", "occupation"])
```

输出写在 `--out` 目录下:

| 文件 | 内容 |
|---|---|
| `pmfg_edges.csv` | 过滤后保留的边(岗位对及其相似度) |
| `communities.csv` | 每个岗位的群落归属 |
| `community_stats.csv` | 群落数量、模块度、群内与群间相似度等 |
| `core_tasks.csv` | 每个群落的任务按覆盖率排序,前 10%/20%/30% 打了标记 |
| `block_decomposition.csv` | 块(2-连通核心)与桥的统计 |
| `edge_homophily.csv` | 边两端同属一个群落/企业/行业的比例,附随机基线 |
| `overview.png` | 一张总览图(安装 matplotlib 后) |

不带数据也可以先看一个合成的例子:

```bash
python -m ocnet demo --out out/demo
```

## 复现与核对

```bash
python tests/run_tests.py                              # 单元测试 + 12 个随机等价性试验
python validation/random_trials.py --trials 204        # 块分解与朴素实现的逐边核对
python validation/tie_order_robustness.py --runs 5     # 并列换序后结果稳定性的检查
```

方法细节、每一步的依据和取舍在 [docs/method.md](docs/method.md);在一份真实劳动力数据上的应用说明与结果在 [docs/application.md](docs/application.md);上面三条命令各自的核对内容和结果在 [docs/validation.md](docs/validation.md)。

## 目录结构

```
src/ocnet/            库(similarity / pmfg / community / structure / pipeline / cli)
tests/                单元测试与随机等价性试验
validation/           等价性核对与换序稳定性脚本,结果以 CSV 存档
examples/synthetic.py 合成数据生成器,可生成带群落结构的岗位—任务对应表
docs/                 方法说明、应用说明、验证说明
```

## 许可

MIT,见 `LICENSE` 与 `NOTICE.md`。
