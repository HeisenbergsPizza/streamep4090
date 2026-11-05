# CPU MoE 调度模拟器

这里运行离散事件/阶段时间模型，使用 CPU 回放路由和查表计算成本；不执行真实模型推理，不需要编译 CUDA 扩展。Python 3.10+，在 `StreamInfer` 目录执行：

```bash
python -m pip install -r simulation/requirements.txt
python -m simulation.run --help
```

## 后续研究实验的配置约束

从2026-10-10起，研究实验以真实推理系统的代码/启动脚本为配置基准。每次明确目标模型、部署拓扑、DP/EP/TP、batch上限、调度参数、准入/工作负载与成本来源，并保存参考路径及hash。除明确研究的变量外，不为缩短仿真或改变结果自行缩小batch上限。参数扫描、消融和测试配置应单独标注。

当前 `steady_experiment` 从 `experiments/scripts/delta/eval/config.sh` **读取** Attention cap、Expert-group cap和Defrag参数：当前为 **A256/E1024、lookahead=4、lookback=4、decay=0.8**。每次 `configuration.json` 的 `runtime_reference` 保存脚本SHA256、参考值、匹配项与显式覆盖；缺少或无法解析参考值时报错，不静默退回小batch。

调度粒度默认 `--expert-scheduling layer`，对应当前真实系统的本地expert group执行粒度。显式设置 `--expert-scheduling expert` 会启用本模拟器的expert-wise研究扩展，结果标记 `simulation_only_extension=true`；该开关不修改其余实验默认参数，也不声称复现了作者尚未实现的TODO算法。

该参考仅覆盖上述五个参数。当前单节点4卡研究设定、跨模型路由/成本组合、输出长度、KV准入以及未实测的Attention/通信成本，仍须在实验计划中逐项核对并标明差异。匹配这五项不能据此声称完整配置或真实GPU性能已经对齐。改变目标系统时应同步更换参考来源。

`simulation.run --synthetic` 的小规模默认值仅用于smoke；使用真实profile时，必须显式提供两个batch cap。旧 `experiment.py` 与低层Python API保留兼容默认值，不能直接将其默认参数当作新研究基线。历史报告保留原配置及结果，后续实验另建目录。

## 先运行合成 smoke

```bash
python -m simulation.run --synthetic --mode async \
  --policies defrag,flfs,mtfs,fifo \
  --gpus 2 --layers 4 --experts 8 --top-k 2 \
  --requests 32 --output-tokens 8 --active-request-cap 16 \
  --attention-batch-cap 16 --expert-batch-cap 64 \
  --arrival-rate 0 --seed 42 --schedule-trace-limit 200 \
  --plot-itl-cdf --output-dir /tmp/streamep-synthetic-async
```

这会生成明确标注 `synthetic` 的随机 Top-k Parquet 和公式计算 CSV，保存到输出目录的 `synthetic-inputs/`。专家成本人为设为 `0.01 + 0.002 × batch_size` 毫秒，只用于检查流程、守恒和策略行为，不能当作任何 GPU 的性能结果。

同一命令中的策略复用相同路由文件、计算成本、到达 seed、请求数和拓扑；仅策略改变。默认跑完整有限工作负载。`--arrival-rate` 单位为**请求/秒**，CLI 转换为请求/tick；0 表示全部请求在时间 0 到达。直接 Python API 的 `arrival_rate` 单位仍是请求/tick。

同步与双微批基线分别执行：

```bash
python -m simulation.run --synthetic --mode sync --output-dir /tmp/streamep-synthetic-sync
python -m simulation.run --synthetic --mode tbo --output-dir /tmp/streamep-synthetic-tbo
```

上述两个命令使用与默认 async smoke 相同的工作负载参数。`--policies` 只对 async 有效。基线用于比较同步阶段/双微批抽象；没有逐 GPU 的统一 ready 队列竞争。TBO 的两个微批共享接收端 FIFO 通道，分别建模同主机与跨主机接收争用，通信可与计算重叠；其计算仍按集中式阶段耗时调度。

## 输入真实采集的 profile

本地 `trace_analysis` 的 `processed_16fields` 缓存可以先转成模拟器 Parquet：

```bash
python -m simulation.sample_trace \
  --cache-dir /absolute/path/trace_analysis/data/processed_16fields \
  --output-dir /tmp/qwen16fields-sample \
  --requests 256 --output-tokens 32 --seed 20261005
```

导出器对每个领域等额、无放回随机抽取请求，再打乱请求顺序；保留每条请求的连续 decode 前缀和原始 Top-k 顺序。它核验缓存与来源 manifest 的哈希，生成 `routing.parquet` 和包含原请求映射的 `sample_manifest.json`，不修改源数据。请求数必须能被领域数整除，输出长度不能超过缓存或 100 个 token。

当前这份 16fields 数据来自 **Qwen3-235B-A22B-FP8，94 层、128 专家、Top-8**。默认导出全部 94 层；显式增加 `--layers 48` 会只保留前 48 层。若搭配 Qwen3-30B 的成本曲线运行，必须标注为跨模型路由代理实验，不能称为真实 Qwen3-30B 路由。模拟器的请求数和输出长度应与导出值一致，避免循环回放凑样本。

没有 `--synthetic` 时，CLI 要求显式提供路由和专家成本：

```bash
python -m simulation.run --mode async --policies defrag,flfs,mtfs,fifo \
  --routing-profile /absolute/path/routing.parquet \
  --expert-profile /absolute/path/expert.csv \
  --attention-profile /absolute/path/attention.csv \
  --gpus 4 --layers 48 --experts 128 --top-k 8 \
  --requests 256 --output-tokens 64 --initial-context 1024 \
  --active-request-cap 128 --attention-batch-cap 256 --expert-batch-cap 1024 \
  --arrival-rate 100 --seed 42 --plot-itl-cdf \
  --output-dir /tmp/streamep-profile-run
```

路径与模型参数是模板，必须按实际文件修改。CLI 不自动采用仓库 A100 CSV，也不会根据文件名认证采集硬件或测量范围。每次输出输入文件的绝对路径和 SHA256；用户提供的 profile 标注为 `provided_profiles_unverified_provenance`。

路由 Parquet 每行表示一个请求的一个输出 token 在一个 MoE 层的 Top-k 选择：

| 列 | 约束 |
| --- | --- |
| `rid` | 非负整数，请求 ID |
| `token_index` | 非负整数，输出 token 索引 |
| `layer` | 层索引从 0 连续，覆盖模拟层数 |
| `expert_logical_k0` … `expert_logical_k(K-1)` | 恰好 K 个专家列；每行互异；ID 在 `[0, experts)` |

`rid`、`token_index` 可以稀疏。每个已记录的请求/token 必须包含每层，重复行、缺层、负数、重复 Top-k 专家和越界 ID 都会报错。回放按排序后的请求/token 位置循环，超过 trace 长度会复用路由；不会重新生成模型 logits，不按“trace 中出现过多少专家”投影专家 ID。输入前应自行过滤到 decoding 阶段；加载器不识别 prefill/decode 标签。

单专家成本 CSV：

```csv
batch_size,avg_time_ms
1,0.012
2,0.014
```

上面的数值仅展示格式。文件必须覆盖 `1..expert_batch_cap` 的所有 batch size，耗时有限且大于 0，不做缺项插值。默认专家组成本是各非空专家查表耗时之和，属于**串行求和代理模型**，没有内置 0.7 等加速倍率。`--expert-cost-scale` 默认 1；修改它属于人为假设。

可选 `--grouped-profile` 使用完整本地专家负载向量的精确查表：

```csv
expert_counts,time_ms
"[2, 0, 3, 0]",0.025
```

向量顺序为当前 GPU 上连续放置的本地专家顺序，长度为 `experts/gpus`，包括零负载槽。必须覆盖实际形成的每个向量；缺少精确测量就报错，不自动估算 grouped GEMM 加速。即使提供此文件，单专家 CSV 仍是必需输入。

可选 `--attention-profile` 在 `simulation.run`（async/sync/TBO）及 `simulation.steady_experiment` 中均可使用。只支持纯 decode，每个请求每步一个 query token。以下三列格式表示**整个 Attention 阶段**的耗时，继续兼容原有总耗时 CSV：

```csv
batch_size,context_length,time_ms
1,1024,0.04
1,2048,0.06
2,1024,0.05
2,2048,0.07
```

数值只展示格式，未做 GPU 测量。成本输入为实际执行批次的 `B = len(contexts)` 和 `L = mean(contexts)`，其中每个 context 为 `initial_context + token_index`，表示当前 query 之前的 KV 历史长度。L 保留小数；不使用最大 context，不对 B 做 padding。精确网格点直接查表，其余在包围 `(B, L)` 的四角上双线性插值；命中某一轴时退化为一维插值。超出任一轴测量范围或缺少所需角点时报错，不外推、不回退到代理。空 batch 返回 0。

二维插值思路参考 [LLMServingSim 2.0 的 pure-decode lookup](https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/serving/core/trace_generator.py#L1011-L1064)。这里保留精确均值，采用严格范围检查，未引入上游的均值取整、外推或 skew correction。均值相同而长度分布不同的 batch 会得到相同估计。

需要分项耗时时，添加 `component` 列，按分项记录完整网格。例如下面将总耗时拆为三个**互不重叠**的计时范围（数值仍仅演示格式）：

```csv
component,batch_size,context_length,time_ms
qkv,1,1024,0.010
attention_core,1,1024,0.025
o_proj,1,1024,0.005
qkv,1,2048,0.010
attention_core,1,2048,0.045
o_proj,1,2048,0.005
qkv,2,1024,0.012
attention_core,2,1024,0.032
o_proj,2,1024,0.006
qkv,2,2048,0.012
attention_core,2,2048,0.052
o_proj,2,2048,0.006
```

各分项独立插值后求和。组件名可自定义，例如额外记录 norm、RoPE、KV 写入；需根据实际 kernel 的融合与计时边界划分，不能重复计算。每个已记录坐标必须包含相同的组件集合；重复行、空组件名、负值、非有限值、总耗时为 0 都会报错。允许单个组件为 0；保留名 `total` 不得和其他组件混用。仅随 B 变化的分项也需在各 context 坐标重复填写；当前不推断缺项或自动补算子。

CSV 单位为 **ms**，加载时乘 `ticks_per_millisecond` 一次。不能直接把其他 profiler 的 `time_us` 改列名后导入。Python 接口 `AttentionCostModel(...).evaluate(contexts)` 返回 `batch_size`、`context_mean`、`components_ticks` 与 `total_ticks`；原有 `model(contexts)` 仍返回总 ticks。所有模式的结果记录 `attention_cost_model` 元数据。async 的 Attention 调度记录额外包含 `attention_context_mean` 和 `attention_components_ticks`，其分项和等于该批计划执行时长；仍按总时长执行一次不可抢占计算，不模拟分项间并发。这些值属于计划成本分解，早停时不能直接当作已经执行的 busy time。

未提供 profile 时，保留 `batch_size × --attention-per-token-us`，默认每 token 20 微秒，并标记 `linear_per_token_proxy`。一张表对应固定模型、设备、dtype、并行切分和 kernel 配置；没有这些实测输入，新增查表能力本身不构成硬件校准。实验前需要覆盖实际可能形成的 B 和 context 均值范围。

采集时应记录 GPU 型号、时钟/功率、dtype、权重/算子形状、kernel 实现、warmup/重复次数、CUDA Graph、缓存状态、计时范围和统计方法。尤其要说明 expert 时间覆盖单次 GEMM，还是完整 up/gate/activation/down 路径。仓库已有 CSV 不能仅凭文件名当成完整 expert forward 测量；现有 `benchmark/ops/moe/expert.py` 可供阅读采集方式，但输出范围须检查后再用。当前仓库未附默认 gating Parquet 的下载或采集流程，需自行提供兼容 trace。

### 已接入的 Qwen Attention profile

`simulation/attention_costs_profiles/qwen3_30b_a3b_rtxpro6000_tp1/attention.csv` 来自 [LLMServingSim 的 Qwen3-30B-A3B-Instruct-2507 bundle](https://github.com/casys-kaist/LLMServingSim/tree/b09c7aaab4b072d73136641841cc2688616c49eb/profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16/tp1)，固定上游提交为 `b09c7aaab4b072d73136641841cc2688616c49eb`。采集设备为 RTX PRO 6000 Blackwell Server Edition，BF16、TP=1，元数据记录 vLLM 0.28.0、CUDA 13.0、dummy weights 和 eager 执行。它提供真实 GPU 算子计时；模型仅保留采集所需层数，不能据此视为完整模型或多卡服务实测。

适配器从上游 `attention.csv` 选择 `prefill_chunk=0`、`prefill_key=0`、`decode_q_len=1`、`n_decode>0` 的 pure-decode 行，以 `n_decode` 和 `kv_decode` 对应 batch size 与 query 之前的 KV 历史长度。共同网格为 **40 个 batch 点 × 19 个 context 点**，覆盖 B=1–256、context=16–16384：保留此范围内全部 dense 实测 batch 节点，Attention core 在新增 batch 节点上做插值。导出表因此包含实测值和明确的插值值；每个查询仍由共享成本模型插值，超范围时报错。上游 `time_us` 除以 1000 转为 `time_ms`，仿真加载时再转换为 ticks。

每层成本包含 QKV projection、Attention core、O projection、RoPE、两次 layernorm 和一次合并的 Q/K norm。两次 layernorm 使用同一条上游测量曲线；`qk_norm` 已合并 Q/K 两部分，不再乘 2。没有将 embedding、最终输出 norm/head、MoE/router、CPU 调度、batch reformation 或通信计入这张表。当前 StreamInfer Attention 实现没有 Q/K norm，因此这里采用上游 Qwen 算子范围，不能称为 StreamInfer 运行时的精确复刻。通信仍由原有独立模型计算，其他未测开销不会由适配器自动补齐。

同目录的 `attention.manifest.json` 记录来源、转换方式和成本边界；`source/` 保留固定版本的 `attention.csv`、`dense.csv`、`meta.yaml`、`LICENSE`、`qwen3.yaml` 和 `timings.py`。在仓库根目录可离线重建到一个新路径：

```bash
python -m simulation.import_llmservingsim_attention \
  --source-dir simulation/attention_costs_profiles/qwen3_30b_a3b_rtxpro6000_tp1/source \
  --output /tmp/qwen_attention.csv
```

通用入口的默认成本保持不变，通过显式 `--attention-profile` 选择这张表。下面使用已有采样 trace，进行 4 个逻辑 GPU 的短验证；需要先在当前 Python 环境安装本 README 开头列出的依赖，并选择尚不存在或为空的输出目录：

```bash
python -m simulation.steady_experiment \
  --routing-profile reports/simulation/qwen16fields_20261005_4gpu_proxy/trace/routing.parquet \
  --expert-profile simulation/expert_costs_profiles/Qwen3-30B-A3B_A100.csv \
  --attention-profile simulation/attention_costs_profiles/qwen3_30b_a3b_rtxpro6000_tp1/attention.csv \
  --output-dir reports/simulation/qwen_attention_profile_smoke \
  --policy defrag --expert-scheduling layer \
  --gpus 4 --layers 48 --experts 128 --top-k 8 \
  --attention-batch-cap 256 --expert-batch-cap 1024 --active-request-cap 1024 \
  --hidden-dim 2048 --bytes-per-element 2 --initial-context 1024 --output-tokens 32 \
  --network-bandwidth-gbps 800 --network-latency-us 5 \
  --rate 50 --duration-s 1 --warmup-s 0.2 --block-s 0.2 \
  --trace-duration-ms 0 --seed 20261014
```

该配置组合 RTX PRO 6000 Attention 微测量、A100 Expert 微测量和既有 Qwen trace，仍是**混合硬件成本的代理研究配置**，结果保留 `hardware_calibrated=false`。活跃请求上限 1024 不代表已验证 KV 显存容量。这个短运行用于验证输入、查表和有限请求排空，不能用于报告稳定吞吐上限或真实 A100/RTX PRO 6000 多卡性能；替换其他 scheduler 时应保持同一成本表、trace 和负载配置。

## 模拟范围与参数

- 可选 `--overhead-model paper` 增加论文 Table 4 的 CPU **Schedule** 暴露开销，以及假定为基础计算时间 5% 的 GPU reformation；默认 `none` 保留原行为。CPU 默认隐藏比例为 Attention 50.6%、Expert 68.3%，GPU reformation 默认不隐藏。5% 是可覆盖的假设，不是论文 Figure 17 的 4.5% 吞吐增益。参数、公式、计费范围、基线边界与敏感性命令见 [OVERHEADS.md](OVERHEADS.md)。
- 可选 `--overhead-model structural` 按批次结构建模：CPU 初值为 `10 + 0.02×输入行数 + 0.1×片段数 + 0.05×队列槽数` us；GPU 重组初值为 `3 + 读写字节数/(500×1000) + 0.05×片段数` us，分别可用 `--cpu-schedule-scale` 与 `--reformation-scale` 缩放。各系数均为可调假设，500 的单位为 **GB/s**，没有硬件校准。片段是 logical buffer 中所选行的连续区间；Expert 单连续输入且顺序兼容时免 GPU 重组，Attention 仍 pack/聚合。两种隐藏比例默认均为 0，仍使用固定比例抽象，没有独立 CPU 资源或 CPU/GPU 流水线。支持 `simulation.run --mode async` 与 steady 的 async/sync/TBO；旧 `simulation.run --mode sync|tbo` 和 legacy 入口拒绝此模式。开启 `--schedule-trace-limit` 可查看逐批工作量与成本分项；完整定义、参数及合成命令见 [OVERHEADS.md](OVERHEADS.md#structural按批次结构计费)。
- 单节点逻辑 GPU；每 GPU 共置 Attention 和所分配的专家，一次只执行一个不可抢占计算批次。专家按连续 ID 均分，Attention 请求 home GPU 轮转分配。
- async 的流水线为 `A0,E0,…,A(L-1),E(L-1),A_L`；各 Top-k 分支异步回传，全部到达后进入下一 Attention 队列。终端 `A_L` 参与调度，`--finalize-per-token-us` 默认 0，表示终端计算成本未计入。
- defrag 对专家副本数除以 Top-k，与 Attention token 数统一比较。得分包含当前量与衰减的后续流水线当前/历史负载；默认 lookahead 4、lookback 4、decay 0.8。lookahead 1 只看当前量，lookback 0 不保留历史。history 在每次成功调度后记录整个 GPU 快照，专家有效量按整数截断；同分按统一索引先全部 A 再全部 E。
- 默认 `layer` 粒度下，FLFS 选择流水线最前的非空槽；MTFS 选择有效 token 数最多的槽；FIFO 选择最早进入 ready 队列的槽。选定专家层后，所有策略共用按 ready time 合并各专家队头的 FIFO 组批，同一时刻按专家 ID 排序；`--expert-batch-cap` 限制该次专家组计算的分支总数。FIFO 选层和这项共同的层内组批规则是两个独立步骤。这些策略均无额外 aging/progress 优先规则；有限请求跑完不代表开放到达下无饥饿保证。
- 同卡传输成本为 0；跨卡成本为 `latency_us/1e6 + token_copies × hidden_dim × bytes_per_element / (Gbps × 1e9 / 8)` 秒。async 使用接收端 FIFO 争用，默认 100 **Gb/s**、5 微秒启动延迟均为假设。没有 NCCL 协议、发送端/交换机争用或真实 PCIe/NVLink 拓扑建模。
- 活跃请求上限只约束请求数量；不模拟 KV 显存容量、prefill、模型权重载入、TP、实际 kernel/通信干扰或 GPU 测量。`--initial-context` 只用于 Attention 成本查表。
- `--tokens-after-saturation N` 可在达到活跃请求上限后再完成 N 个 token 早停。此时未完成请求属于截断，不能把完成 token/固定输出长度当成真正完成请求数。

### Expert-wise 调度扩展

这里实现的是现有统一调度器的模拟器扩展；作者统一运行时尚未实现该路径，因此不能将此公式称为作者 TODO 的精确复现。

`simulation.run --mode async` 与 `simulation.steady_experiment` 均支持 `--expert-scheduling expert`；直接调用 `sim_async.run_simulation` 时传入 `expert_scheduling="expert"`。sync/TBO 不支持该扩展，CLI 会在创建输出目录前拒绝。`layer` 仍是默认值，保持原有层内合并组批行为。

expert模式中，每个GPU的候选集合是 `A_0,…,A_L` 和每层每个本地专家的 `E_(l,e)`。Defrag对整个集合取最高分，直接选定一个Attention队列或一个专家队列；不会先选整层再从层内选专家。专家候选的当前量为该专家ready分支数除以Top-k。其lookahead沿原来的 `A_l → E_l → A_(l+1)` 阶段流水线前进，后续expert阶段的占用仍为该层所有本地专家ready分支数之和除以Top-k，避免假设同层专家之间存在前后依赖。

具体地，令 `c` 为候选队列，`s(c)` 为其Attention/Expert阶段，`q(c)` 为候选当前有效量，`Q(s)` 为阶段当前有效量，则

```text
score(c) = q(c) + sum[k=1..lookahead_steps-1] decay^k * (Q(next^k(s(c))) + H_B(next^k(s(c))))
```

其中，Attention的 `q` 为ready token数，expert的 `q` 为单专家ready分支数/Top-k；`H_B` 保持原实现的阶段历史快照均值，expert阶段先按Top-k归一化并整数截断。快照在成功选择后保存选择前的队列状态，历史窗口与lookahead窗口独立；流水线延续原实现的循环语义。当前候选项不加自身历史。同一expert层的候选共享后续项，但各自当前量不同，因此最满的专家不一定是全GPU最终得分最高的队列。

FLFS、MTFS、FIFO也使用同一个expert候选集合：FLFS先按阶段先后选取，同一阶段按expert ID；MTFS比较每个候选的当前有效量；FIFO比较每个候选队头的ready time。Defrag/MTFS同分先按全部Attention、再按expert的层号与本地expert ID排序；同一expert队列内部保持FIFO。

选中 `E_(l,e)` 后，只弹出该队列最多 `--expert-batch-cap` 个分支，其余专家队列保持不变。该cap在expert模式下约束一次单专家计算的分支数，数值默认仍跟随真实系统参考；各expert同时独立拥有同样的cap并不表示它们同时执行。每GPU仍只有一个不可抢占计算批次，没有新增同卡kernel并行。成本模型接收只有选中expert非零的负载向量；使用精确grouped profile时也必须提供相应向量，缺项继续报错。

例如在前述异步命令中增加 `--expert-scheduling expert`，并换用新的输出目录，即可在相同输入、cap及成本假设下检验单专家粒度。比较性能时应同时检查批量大小、ready等待、Top-k join等待和完整排空情况；本扩展不增加额外aging或进度保证。

## 输出与指标

每次运行产生 `inputs.json`、每策略 JSON、每策略逐 token CSV 和 `comparison.csv`。`--plot-itl-cdf` 额外产生 `itl_cdf.png`。输出目录必须不存在或为空；非空目录会报错，防止覆盖既有实验。未指定目录时自动在 `simulation-results/` 下生成时间戳目录。

async JSON 含有 ready wait、join gap、计算 busy fraction、batch histogram、完成/待处理状态，以及最多 `--schedule-trace-limit` 条决策。终端 finalize 的 ready wait 和 batch histogram 独立统计，不计入 Attention 的对应统计。TBO 的该参数控制饱和后的微批时间线；sync 不输出统一调度决策。

输入配置、结果摘要及steady对比配置均记录 `expert_scheduling`；`provenance.scheduling`（通用CLI）或 `configuration.scheduling`（steady）记录真实系统的layer粒度、是否属于模拟器扩展及cap作用范围。研究结果不能仅凭五项参考参数一致就把expert模式归类为真实系统调度。

### 追踪 scheduler：队列热图与执行时间线

在 async 命令中增加 `--queue-trace`，即可记录完整观察窗口内的 ready 队列和调度决策。例如：

```bash
python -m simulation.run --synthetic --mode async --policies defrag,flfs,mtfs,fifo \
  --gpus 2 --layers 8 --experts 8 --top-k 2 \
  --requests 32 --output-tokens 16 --active-request-cap 16 \
  --attention-batch-cap 16 --expert-batch-cap 32 \
  --queue-trace --queue-trace-interval-ms 0.1 \
  --output-dir /tmp/streamep-scheduler-trace

python -m simulation.plot_scheduler_trace \
  --traces /tmp/streamep-scheduler-trace/async_defrag_queue_trace.npz \
           /tmp/streamep-scheduler-trace/async_flfs_queue_trace.npz \
           /tmp/streamep-scheduler-trace/async_mtfs_queue_trace.npz \
           /tmp/streamep-scheduler-trace/async_fifo_queue_trace.npz \
  --gpu 0 --output-dir /tmp/streamep-scheduler-figures
```

真实路由实验继续使用前述 `--routing-profile` 等参数；追踪器仅观察，不新增模拟事件、不调用随机数、不修改组批与调度策略。`--queue-trace-start-ms`（默认 0）和 `--queue-trace-end-ms` 可限制记录范围，模拟仍按原停止条件继续。观察器开关的结果一致性有独立测试。

每个策略额外输出三个文件：

- `async_<policy>_queue_trace.npz`：时间边界与每个 GPU、每个阶段/层的 ready 队列深度。
- `async_<policy>_queue_trace.metadata.json`：维度、单位、请求/实际观察窗口、文件路径和统计口径。
- `async_<policy>_queue_trace.decisions.jsonl.gz`：与观察窗口相交的完整执行记录（含窗口开始前启动的执行），含 GPU、阶段、层、起止时间、batch size、专家负载向量、决策时的队列快照与候选分数。它不受 `--schedule-trace-limit` 限制。结束时间是计划完成时刻；提前停止时，绘图会裁剪到实际观察范围。

expert模式的执行记录额外保存全局 `expert_id`、`local_expert_id` 和 `selected_queue_index`。决策快照及分数展开为 `[A_0,…,A_L,E_(0,0),…,E_(0,m-1),E_(1,0),…]`，其中 `m=experts/gpus`；Attention执行的expert ID为空。时间加权NPZ热图仍按层聚合全部expert，执行图纵轴也仍表示层号，不能从纵轴单独判断选中了哪个expert；具体选择应查看决策中的expert ID和负载向量。

热图每个像素保存 **该时间 bin 内队列深度的时间平均**：`integral(Q(t) dt) / observed_duration`。积分由每次入队/出队更新，覆盖算子执行期间的新到达；不是把不等间隔的调度快照插值或按决策次数平均。同一时刻的瞬时入队/出队不占时间。最后一个 bin 使用实际覆盖时长，未观察的时间不补成零。

绘图以策略为行、Attention/Expert 为列，默认显示所有模型层。Attention 为 ready token 数，Expert 默认除以 Top-k 显示有效 token 数（原始副本数保存在 NPZ）；不同策略共享颜色标尺。终端 `A_L` 在原始数据中独立保留，层热图与非空模型层统计均排除它。执行时间线独立显示实际运行区间，避免把排队误认成计算；零成本 finalize 仍保留在决策日志。

另有非空 ready 层数随时间图和各 GPU 的窗口统计。这里的“非空层”表示 `A_l > 0 OR E_l > 0`，不包括正在计算的、通信中的或尚未收齐 Top-k 的分支。该指标描述 ready 工作的分布，不能单独当作所有在途请求的碎片化指标。选择 `--start-ms/--end-ms` 比较局部窗口时，应使用相同绝对时间范围；绘图会报告实际采用的 bin 边界。仅凭热图不能认定某策略达到更高真实 GPU 利用率。

逐 token CSV 对全部完成 token 写出记录。`itl_ms` 为同一请求相邻 token **完成时刻差**，排除 `token_index=0`；`token_execution_latency_ms` 为出生到完成的时间，两个定义在基线中可能不同。首 token 不是包含 prefill 的 TTFT。Attention ready wait 累加 token 各 Attention 层等待，终端队列等待单独记入 `finalize_ready_wait_ms`；expert ready wait 是各分支等待之和，不能当作该 token 的关键路径时间；join gap 是每层首个到末个分支回传的间隔之和。

`comparison.csv` 的 ready wait / join wait 均值沿用 async 事件统计：Attention 每次进入计算、expert 每个分支进入计算、join 每层汇合各为一个样本；它们与逐 token CSV 的跨层累计值具有不同分母。相关 `metrics_ms` 中保存样本计数，早停时可能包括尚未完成 token 的已处理阶段。

吞吐同时报告 output tokens/s、真正完成 requests/s，以及完成 token 数除以固定输出长度的 equivalent requests/s。分母是从时间 0 到模拟结束的 makespan，包含到达间隙和排空阶段。busy fraction 是该时间窗内 GPU 被模拟计算占用的比例，不等价于实际 SM utilization。

计数、均值、最大值使用所有完成样本；分位数和 CDF 最多使用 `--metric-sample-capacity` 个固定 seed reservoir 样本，JSON 记录样本数和是否抽样。没有完成样本的指标为 null。基线未提供的 ready wait/utilization 字段留空，避免填入虚构的零。

## 持续到达与稳定负载边界

`steady_experiment` 使用固定时间范围的 Poisson 请求到达；到达停止后继续排空。它以完整请求为单位有放回重采样 routing profile：同一请求的 token 顺序、层和 Top-k 保留，显式校验输出长度不超过源请求，禁止把短 token trace 绕回续接成长请求。相同 rate、seed、profile 在各策略间生成字节相同的 `workload.csv`。

### 逐 token 的跨层推进追踪

在 `simulation.steady_experiment --mode async` 命令中追加 `--token-trace`，可将详细队列观察窗口内选择的每个 Attention batch 作为一个 cohort，追踪其成员经过 Expert 分支返回、Top-k 齐备，直到下一阶段 Attention 开始。窗口沿用 `--trace-duration-ms`：默认 40 秒到达、5 秒 block、200 ms trace 对应 `[35.0,35.2)` 秒。窗口只限制 source Attention 的选择；已选 cohort 的后续事件即使超出窗口也继续记录。

输出前缀为 `token_progress_trace`：`.tokens.jsonl.gz` 保存逐 token-layer 的源 batch、路由专家、各分支计算/返回时间、join ready 与下一 batch；`.batches.jsonl.gz` 保存相关 batch 的完整成员及计算边界；`.metadata.json` 保存采样口径、完整性与未完成记录计数。终端 `finalize` 可以作为最后一层的后继，分析真实跨层推进时应单列它。

该功能只增加观察记录，不增加模拟事件或修改随机数与调度。`selection_ms` 与 `gpu_start_ms`、`compute_start_ms` 分开保存：启用额外 overhead 时，后两者包含相应 CPU 暴露/重组成本。成本分项的时间边界是模型的计划分解，实际 batch 完成另有 `actual_completion_ms`；它们均非 GPU kernel 实测。提前停止或异常留下的 cohort 会标记为 censored，不能当作完整样本。

```bash
python -m simulation.steady_experiment \
  --routing-profile /absolute/path/routing.parquet \
  --expert-profile simulation/expert_costs_profiles/Qwen3-30B-A3B_A100.csv \
  --policy defrag --rate 70 --duration-s 40 --warmup-s 10 --block-s 5 \
  --seed 20261007 --output-dir /tmp/steady-study/defrag_rate70_seed20261007

python -m simulation.steady_analysis \
  --input-dir /tmp/steady-study --output-dir /tmp/steady-study-analysis
```

上例默认延续4逻辑GPU/48层/128experts/Top8/每请求32输出/active cap128；这些仍是待逐项核对的研究设定。Attention cap和expert-group cap现从真实评测配置读取，当前为256/1024；Attention20us/token、800Gbps与5us通信仍是假设。代码、参考配置和输入hash随每次实验保存，具体值见 `configuration.json`；参数来源一致不代表成本已经硬件校准。

steady命令增加 `--expert-scheduling expert` 可运行单专家粒度研究，其余参考默认值不变；`runtime_reference.scheduling` 显式标记这一调度扩展。为每种粒度使用独立输出目录，并保留相同的路由、seed、到达率、批量上限和成本来源。

### 使用相同观察器比较同步 EP 与 TBO

`steady_experiment --mode async|sync|tbo` 使用共同的事件驱动入口、Attention/Expert 成本函数和接收端 FIFO 资源。默认 `async` 保持原有队列调度，并要求指定 `--policy defrag|flfs|mtfs|fifo`；`sync` 和 `tbo` 的 policy 分别记为 `sync`、`tbo`，可省略 `--policy`。两个基线只接受 `--expert-scheduling layer`，错误组合会在创建输出文件前被拒绝。

同步 EP 每个 decode iteration 使用一个 collective batch，TBO 将该 iteration 分成两个 microbatch，通过通信与另一微批的计算重叠推进。基线保留阶段屏障和 iteration 边界准入；全局同一时刻只允许一个 collective compute phase，各 GPU 在该 phase 内串行执行本地批次，提前完成本地计算后的屏障等待不计作计算 busy time。请求的外部到达仍发生在 `workload.csv` 的原始时间，不能将等待下一个 iteration 的准入延后误写成到达延后。TBO 不表示同一 GPU 同时执行两个计算 kernel，也没有增加硬件并发或削减 profile 成本。

例如沿用当前研究的输入、cap 与测量窗口运行同步 EP：

```bash
python -m simulation.steady_experiment --mode sync \
  --routing-profile reports/simulation/qwen16fields_20261005_4gpu_proxy/trace/routing.parquet \
  --expert-profile simulation/expert_costs_profiles/Qwen3-30B-A3B_A100.csv \
  --attention-profile simulation/attention_costs_profiles/qwen3_30b_a3b_rtxpro6000_tp1/attention.csv \
  --gpus 4 --layers 48 --experts 128 --top-k 8 --output-tokens 32 \
  --initial-context 1024 --active-request-cap 1024 \
  --attention-batch-cap 256 --expert-batch-cap 1024 \
  --rate 200 --duration-s 40 --warmup-s 10 --block-s 5 --seed 20261011 \
  --output-dir /tmp/steady-sync-ep-r200
```

改为 `--mode tbo` 并使用新输出目录即可运行 TBO；与 async 配对时保持其余输入、到达 seed、profile 和窗口相同。配置中的 `execution_mode`、`scheduling` 和代码哈希分别记录执行模式、微批数、准入规则和实现身份。Defrag 的三个评分参数在 baseline 中不参与决策，虽然参考值仍会被保存。这些 collective 模型是仿真基线，参考脚本的五项参数匹配不构成其与真实运行时的执行等价证明。

三个模式均输出下述 `requests.csv`、`tokens.csv`、`steady_metrics` 和 `scheduler_queue_trace.*`，可用同一个 `plot_scheduler_trace` 绘图。Baseline 的计算记录表示按 collective 依赖启动的实际本地批次，不应为其虚构候选分数或检查 Defrag argmax。队列图显示数据已就绪且尚未执行的工作；baseline 中这些工作还可能受 collective gate 限制。应联合执行区间、准入等待和模式元数据解释，不能将所有非空队列都视为此刻可自由选择的候选。同步等待、网络在途和未完成的 Top-k join 也不应填进计算占用。

这里的统一观察入口不同于前文 `simulation.run --mode sync|tbo` 保留的旧阶段预算 API；旧入口仍用于兼容已有脚本，不能把其有限的 timeline/指标当成新的 steady 队列记录。跨历史报告比较时需检查代码哈希及模式，而不只检查 profile 文件名。

每个case保存 `workload.csv`、逐请求 `requests.csv`、逐token `tokens.csv`、`result.json` 和观察窗口的NPZ/完整决策gzip。`steady_metrics` 在原事件之间精确积分，不加入采样事件、不使用调度RNG；每100ms记录 active、外部waiting、ready、join、网络分支、尚未完成的输出工作量与逻辑GPU计算占用。默认最后一个5秒测量块开始后的200ms保存细粒度队列/决策，沿用 `plot_scheduler_trace` 绘图。`--trace-duration-ms 0` 可关闭该细节记录。

瓶颈分析还可使用每个窗口及时间bin中的 `gpu_stage_busy_ms/fraction`，将计算占用分为 Attention、Expert 和 finalize；三项之和等于该GPU总计算忙时。`active_cap_full_fraction` 是活跃请求数达到准入上限的时间比例。`receive_busy_fraction` 按 intra/inter、每GPU记录接收FIFO资源的服务占用，`receive_queue_messages_mean` 记录等待该资源的平均消息数；资源占用不等同于真实 NVLink 利用率。直接 API 使用固定通信延迟、未建模接收争用时，这些接收指标为 `null`，模式标记为 `fixed_delay_no_contention`。上述忙时均按实际跨窗口重叠时间裁剪，既不把预热或drain算入测量窗，也不把窗口开始前启动的计算漏掉。

吞吐仅用 `[warmup, arrival_end)` 计数和该窗口时长计算，启动与停止到达后的drain单独报告。ITL/请求延迟按完成时刻归属，ready wait按开始服务时刻归属，准入等待按准入时刻归属；请求日志保留到达/准入/完成时间供另做cohort分析。分支等待保留精确count/sum/mean/max，不伪造分位数；窗口ITL和请求级分位数使用全部窗口样本。完成样本的延迟不覆盖尚未完成请求，因此同时记录最老未完成/待准入请求年龄。

稳定性检查使用 `outstanding = active + waiting = arrived - completed_requests`，不能只看被active cap限制的内部队列或GPU占用。默认按5秒块计算backlog及waiting的全窗、首末块和后半窗漂移，容忍度为 `max(0.25 req/s, 实测到达率×2%)`。无明显漂移归为 `stable`，持续backlog增长且外部waiting增加归为 `overload`，其他归为 `inconclusive`。完成率缺口与backlog净增长是同一守恒量，不当作独立证据重复计票。

20秒/5秒warmup仅用于初筛；默认需要至少40秒到达、10秒warmup、30秒测量，以及3个不同seed一致的结果，才把rate点标为confirmed。分析报告最高已确认稳定rate与更高的已确认超载rate，保留混合seed、灰区与非单调结果。这个区间是当前trace池、成本、准入上限及观察时长下的经验边界；有限运行不能证明数学稳定性或真实GPU容量。每个case最终能排空也不意味着持续到达时稳定。

## 验证

```bash
python -m unittest discover -s tests/simulation -v
```

测试和合成 smoke 验证事件依赖、队列选择、profile 契约及输出一致性；目标硬件校准需要独立采集和验证。

旧 `experiment.py` 保留原有扫描和文本输出兼容入口，并在结果目录的 `run-details/*.json` 保存每个配置的参数、策略、seed、profile 路径和完整研究指标。新实验建议使用本页 CLI，显式传入数据、成本和规模，并保存完整参数与来源。
