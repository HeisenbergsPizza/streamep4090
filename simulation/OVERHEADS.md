# GPU batch reformation 与 CPU 调度成本

这个模式为每个实际执行的非空 Attention/Expert batch 增加 CPU scheduling 和 GPU batch reformation 的暴露开销。目的是做调度策略的敏感性研究；它没有用 4090 实测校准，也没有实现独立 CPU 线程、metadata 搬运资源或真实的 CPU/GPU 流水线。

默认 `--overhead-model none` 保留原有计算成本与执行行为。`paper` 使用论文 CPU Schedule 与假定的 GPU 计算时间比例；`structural` 按实际选择的输入行、连续片段、调度队列和激活字节数计费。新增 `structural` 不改变 `none`、`paper` 的既有行为。三个模式均不是真实硬件性能预测。

## paper：论文数据和假设的边界

来源：[StreamEP 论文](https://doi.org/10.1145/3830418.3843869)，PDF 第 13 页 Table 4。第 19 页 Figure 20 和第 20 页 Appendix E 说明，这组数据来自 Sphere 集群上的 GPT-OSS-120B BF16；Sphere 使用 16 张 L40S。这不是 Qwen-30B 或其他 GPU 的校准结果。

本模式只使用 Table 4 的 **Schedule** 一行：

| 每个非空 batch | CPU Schedule | 被 GPU 执行隐藏的比例 | 默认暴露成本 |
| --- | ---: | ---: | ---: |
| Attention | 127.2 us | 50.6% | 62.8368 us |
| Expert | 29.6 us | 68.3% | 9.3832 us |

Attention 的 67.0% 和 Expert 的 76.2% 是 **CPU total** 的隐藏比例，不能直接套在 Schedule 一行。本模式没有另外加入 Prepare、Launch、Post-process 或 CPU total，避免把 scheduling 与完整 CPU loop 混淆。表中的时间是每批平均值，不是每 token 时间，也没有按批量、层数或 scheduler 类型缩放的实测函数。

论文没有给出 GPU reformation kernel 的独立耗时或其占计算时间的比例。因此，`paper` 模式将 GPU reformation 的原始成本暂定为该批基础计算成本的 **5%**，并在输出中标记为假设，可以显式改成 0 或其他非负比例。

Figure 17 的 **4.5% 是 batch reformation 优化与 CPU/GPU overlap 联合带来的吞吐提升**，不能换算成 reformation kernel 的成本比例。这里的 5% 与该 4.5% 没有数值推导关系。Table 4 的 GPU execution 178.4/437.3 us 也没有用于替换现有计算 profile。

## paper 公式与共同的事件含义

令 `C` 为该批原有 Attention/Expert 成本（已经应用原有 profile 和 expert cost scale），`S` 为上表对应的 CPU Schedule，`a` 为 CPU scale，`h_cpu` 为 CPU overlap，`r` 为 reformation/compute 比例，`h_reform` 为 reformation overlap：

```text
cpu_raw             = a * S
cpu_effective       = cpu_raw * (1 - h_cpu)
reformation_raw     = r * C
reformation_effective = reformation_raw * (1 - h_reform)
total               = C + cpu_effective + reformation_effective
```

先把微秒转换为模拟 ticks，再将这些暴露开销加入当前批次的执行关键路径。`overlap` 是用户指定的平均隐藏比例，不会根据相邻 batch 动态估计，也不会为被隐藏的 CPU 部分创建独立执行资源；首批和孤立批次也沿用这个比例，不能把它解释为逐批实际发生的重叠。网络通信仍沿用原有异步事件与资源规则，可以与新增延迟自然交叠。

默认 `h_reform=0`：论文 §4.5/Figure 9 描述 CPU 为下一批准备 metadata 时与前一批计算重叠，随后 GPU 执行下一批 reformation 再计算。当前模型因此默认将 GPU reformation 完整计入关键路径。显式提高 `--reformation-overlap` 是额外的研究假设。

开销按实际调度到的本地 batch 收取。Expert layer 模式的一次 capped expert group 收一次；expert-wise 模式每次单专家 batch 收一次。这使 batch 碎片化和执行次数自然影响累计开销。不同 scheduler 使用同一组成本参数，不能仅因为策略不同就给某个策略不同的隐藏比例，除非该差异本身就是单独研究的变量。

空批次和终端 `A_L`/finalize 不收这两项开销；原有 `--finalize-per-token-us` 的语义保持不变。普通模型层即使基础计算成本很小，`paper` 仍收固定的 CPU Schedule 暴露成本；比例型 reformation 会随计算成本变化。因此 `paper` 主要研究固定调度开销与比例型重组开销的影响，不能解释同等计算量下不同碎片数、hidden size、Top-k 和 buffer 布局的实际差异。输入计算 profile 应排除这里另外收取的操作，避免重复计费。

## structural：按批次结构计费

这个模式用于研究异步收益的边界：保持单位成本相同，让各执行方式实际形成的批次、片段和数据量影响总开销。全部初值都是可扫描的研究假设，没有由论文或 4090 测量推导，也没有拟合 `paper` 模式的时间。默认 CPU 与 GPU reformation 隐藏比例均为 **0**。

对每个非空本地 Attention/Expert batch，定义：

| 符号 | 代码字段 | 含义 |
| --- | --- | --- |
| `B` | `batch_size` | 输出行数；Attention 是 token 数，Expert 是路由分支数 |
| `N` | `input_rows` | 输入行数；Attention 为各 token 实际返回分支数之和，Expert 为所选分支数 |
| `F` | `fragments` | 各 logical source buffer 中，被本批选中的行形成的连续区间总数 |
| `Q` | `candidate_queues` | 本次选择检查的逻辑队列槽数，定义见下文 |
| `H`、`d` | `hidden_dim`、`bytes_per_element` | 一行激活的宽度与每元素字节数 |
| `V` | `traffic_bytes` | 潜在读写流量 `(N + B) × H × d` 字节 |

原始成本以微秒计算：

```text
cpu_raw_us         = cpu_schedule_scale × (c0 + cN × N + cF × F + cQ × Q)
reformation_raw_us = reformation_scale × (r0 + V / (bandwidth_GB_s × 1000) + rF × F)
```

初始化为 `c0=10 us`、`cN=0.02 us/row`、`cF=0.1 us/fragment`、`cQ=0.05 us/queue`，以及 `r0=3 us`、`bandwidth=500 GB/s`、`rF=0.05 us/fragment`。这些值提供微秒级固定成本与随数据量增长的起点，适合做数量级扫描，不表示已验证的合理硬件区间。`GB/s` 是十进制**字节**带宽，与网络参数 `--network-bandwidth-gbps` 的 **Gb/s（比特）**不同。

CPU 抽象覆盖选择批次与处理批次元数据：固定部分、逐输入行处理、逐片段处理和队列检查。它不代表完整的 CPU Prepare 阶段，不另外声称覆盖 CUDA Launch、Post-process、操作系统调度或独立通信线程。Attention 的 `N` 包含所有 Top-K 输入，因此该项会随分支数增加。`Q` 在取走 batch 前记录：layer 调度为全部 Attention 与 Expert layer 槽，即 `2L+1`；expert-wise 为 `L+1` 个 Attention 槽加当前非空 expert 队列数；共享事件实现的 sync/TBO 控制器已指定阶段，取 `Q=1`。这些是可解释的工作量代理，不是 Python/C++ 指令计数，也不精确刻画不同策略的 lookahead/history 算法复杂度。每个实际非空 batch 收费一次，空轮询与 finalize 暂不收费。

片段追踪采用 logical buffer ID 和行号，不创建真实张量。Forward 中，同一生产 Attention 批次发往每个目标 GPU 的行构成一个 logical buffer，行号按 expert 排序；return 中，同一 Expert 输出批次发往每个 home GPU 的行构成新 buffer，不能继承 Expert 输入 buffer。Collective 可能把多个受 batch cap 限制的计算批次合并到同一通信消息；接收时仍按生产批次 `output_batch_id` 分组，分别分配 buffer 和行号，不因消息合并就获得免费的连续输入快速路径。因此 `F` 不是简单的消息数量。

Top-K join 传递实际返回行的引用。比如同一 buffer 选中行 `[0,1,4,5]` 是两个片段，不会因为 buffer ID 相同就算作连续输入。batch cap 留下的行保留来源，后续选择重新计算片段数。首层/每个 decode token 的初始激活暂设为独立单行 buffer，这是输入布局假设，不是 tokenizer 的真实批次布局。

Expert 输入只有一个连续片段，且按 expert 输出所需的行顺序仍连续时，走快速路径，GPU reformation 成本为 **0**；CPU 成本仍收取。Attention 始终按 pack/Top-K 聚合计费，没有单片段快速路径。`V` 表示潜在读写量，快速路径仍保留该 workload 描述；实际计费流量见 `batch_overhead_components.executed_traffic_bytes`，此时为 0。该模型不显式模拟地址表传输、padding、缓存或 Top-K 加法指令。

选批后仍按固定隐藏比例计算暴露成本，先执行 CPU 暴露延迟，再执行 GPU 重组与基础计算；GPU busy 不包含 CPU 延迟。调整 `--cpu-schedule-overlap` 或 `--reformation-overlap` 仍是人为假设，不能得到 CPU 饱和或真实流水线的预测。后续若加入每 GPU 一个逻辑 CPU 工作线程、有限准备队列和 GPU 完成依赖，可复用上述原始成本函数；届时应由执行时间线产生隐藏比例，停止对同一工作重复应用固定折扣。

开启 `--schedule-trace-limit` 后，每批 trace 除原有 `batch_overhead` 时间外，新增 `batch_workload` 和 `batch_overhead_components`，保存 `N/F/Q/H/d`、流量、各成本分量及快速路径是否命中，便于解释扫描结果。即使不保留逐批 trace，结果的 `batch_overhead_accounting.by_gpu[].by_stage_workload` 仍累计输入/输出行、片段、队列检查、流量与快速路径批次数；这些是已调度批次的工作量，提前停止时可能包含尚未完成的批次。

先固定输入与 CPU 参数，将 `--reformation-scale` 分别设为 `0`、`1`、`2`，可比较无 GPU 重组成本、上述初值和两倍成本。`--cpu-schedule-scale` 独立缩放 CPU 成本；两种 scale 同时为 0 构成零开销对照，应恢复 `none` 的执行时序，结果仍保留结构工作量描述。

在 `StreamInfer/` 下可运行以下合成检查；它验证逻辑流程，不提供真实 GPU 结果：

```bash
python -m simulation.run --synthetic --mode async --policies defrag \
  --gpus 2 --layers 4 --experts 8 --top-k 2 \
  --requests 32 --output-tokens 8 --active-request-cap 16 \
  --attention-batch-cap 16 --expert-batch-cap 64 \
  --hidden-dim 2880 --bytes-per-element 2 \
  --overhead-model structural --reformation-bandwidth-gb-s 500 \
  --schedule-trace-limit 10000 --output-dir /tmp/streamep-overhead-structural
```

## 参数

`simulation.run` 和 `simulation.steady_experiment` 接受相同参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--overhead-model` | `none` | `none`、`paper` 或 `structural` |
| `--cpu-schedule-scale` | `1` | 非负系数，缩放原始 CPU 成本；`paper` 为 Table 4 Schedule，`structural` 为上述四项之和 |
| `--cpu-schedule-overlap` | 未指定 | `paper` 默认 Attention 0.506、Expert 0.683；`structural` 默认 0；显式指定时两类统一使用该 `[0,1]` 比例 |
| `--reformation-compute-ratio` | 未指定 | `paper` 解析为假定的 `0.05`，`none` 为 `0`；显式值必须非负 |
| `--reformation-overlap` | `0` | GPU reformation 的假定隐藏比例，范围 `[0,1]` |

以下参数仅适用于 `structural`；不显式传入时采用表中假设值：

| 参数 | 假设初值 | 单位 |
| --- | ---: | --- |
| `--cpu-fixed-us` | 10 | us/batch |
| `--cpu-per-row-us` | 0.02 | us/input row |
| `--cpu-per-fragment-us` | 0.1 | us/fragment |
| `--cpu-per-queue-us` | 0.05 | us/queue |
| `--reformation-fixed-us` | 3 | us/batch |
| `--reformation-bandwidth-gb-s` | 500 | GB/s，十进制字节 |
| `--reformation-per-fragment-us` | 0.05 | us/fragment |
| `--reformation-scale` | 1 | 无量纲，非负；0 用于取消 GPU 重组成本的消融 |

所有数字必须有限，时间系数非负、带宽严格为正；非默认开销参数要求启用相应模式，不会在 `none` 下静默忽略。`structural` 不接受非零的 `--reformation-compute-ratio`，以免同时使用两个 GPU 成本函数；结构系数也不能传给 `paper`。CLI 在生成输出目录和输入 fixture 前验证配置；结果中保存原始参数和解析后的开销描述。`simulation.run` 写入 `provenance.cost_assumptions.batch_overhead`；steady 写入 `configuration.overhead_config`、`configuration.cost_assumptions.batch_overhead`，并将其纳入 `comparison_configuration` 和代码哈希。比较策略时保持这些字段一致；扫描开销参数时，每组假设单独分析，不能将不同假设混成一条容量曲线。

## 示例与敏感性扫描

在仓库根目录 `StreamInfer/` 执行。先用合成输入检查流程：

```bash
python -m simulation.run --synthetic --mode async \
  --policies defrag,flfs,mtfs,fifo \
  --gpus 2 --layers 4 --experts 8 --top-k 2 \
  --requests 32 --output-tokens 8 --active-request-cap 16 \
  --attention-batch-cap 16 --expert-batch-cap 64 \
  --overhead-model paper --reformation-compute-ratio 0.05 \
  --output-dir /tmp/streamep-overhead-paper
```

使用新目录将 `--overhead-model paper` 改为 `none`、去掉显式 reformation 比例，可以得到原模型对照。只检查 CPU scheduling 时使用 `paper --reformation-compute-ratio 0`；只检查 reformation 时使用 `paper --cpu-schedule-scale 0 --reformation-compute-ratio 0.05`。

下面扫描只是示范性的研究区间，不表示论文证明 50%–80% overlap 或 0%–10% reformation 比例适用于某个硬件。先保留 Table 4 两类各自的 overlap 跑一组，再做统一比例扫描：

```bash
for overlap in 0.5 0.65 0.8; do
  for ratio in 0 0.02 0.05 0.1; do
    python -m simulation.run --synthetic --mode async \
      --policies defrag,flfs,mtfs,fifo \
      --overhead-model paper --cpu-schedule-overlap "$overlap" \
      --reformation-compute-ratio "$ratio" \
      --output-dir "/tmp/streamep-overhead-sweep/cpu${overlap}_reform${ratio}"
  done
done
```

真实 profile 的研究实验继续保留原模型、路由、batch cap、到达率、seed 和观察窗口，在原命令上增加这些开销选项即可。除吞吐和 ITL，还应比较执行批次数、batch 分布、ready wait、join wait 与是否完整排空；结论应描述在这些假设范围内是否稳定。

## 同步基线与旧入口

`structural` 支持 `simulation.run --mode async`、`simulation.steady_experiment --mode async|sync|tbo` 和 `sim_async.run_simulation(execution_mode="async"|"sync"|"tbo", overhead_config={"preset": "structural"})`。API 调用同时传入研究配置的 `hidden_dim` 与 `bytes_per_element`。这些路径保留逐批 logical buffer 来源，按各模式实际选中的输入计算工作量。旧 `simulation.run --mode sync|tbo` 与 `sim_sync.py` / `sim_tbo.py` 入口缺少相同来源信息，会拒绝 `structural`，避免用猜测的片段数计费；做结构成本的同步对照请用 steady 的共享事件基线。

新的 steady async/sync/TBO 模式共享开销函数，按各自实际产生的非空本地 batch 计费。给 sync/TBO 同样的 CPU scheduling 和 GPU reformation 参数，是一种受控成本假设，不能据此声称传统 EP 的实际 reformation 与 StreamEP 相同。传统 EP 与异步组批的实际数据布局和控制流程可能不同，论文也没有提供对齐的基线开销测量。

`simulation.run --mode sync|tbo` 和旧 `simulation/experiment.py` 的 sync/TBO 路径继续使用旧阶段预算实现：暴露 CPU 延迟与 reformation 进入阶段/compute lane 的预算，没有单独 CPU 资源；其阶段占用不能解读成纯 GPU 计算忙时。它们与 `steady_experiment` 的逐 GPU 事件实现有不同执行语义，跨入口不能只凭参数相同就认为时间线一致。`simulation/experiment.py` 的 async 路径使用共享事件实现，也会传递该实验的激活宽度；结构系数扫描优先使用参数齐全的两个新 CLI。

旧实验脚本支持等价环境变量：`OVERHEAD_MODEL`、`CPU_SCHEDULE_SCALE`、`CPU_SCHEDULE_OVERLAP`、`REFORMATION_COMPUTE_RATIO`、`REFORMATION_OVERLAP`，可选 overlap/ratio 留空时采用上述默认。每次配置的解析结果和来源保存于 `run-details/*.json`，例如在原有实验环境设置后执行：

```bash
OVERHEAD_MODEL=paper REFORMATION_COMPUTE_RATIO=0.05 \
  SIM_OUTPUT_DIR=/tmp/streamep-legacy-overhead \
  python simulation/experiment.py async
```

该命令保留旧脚本的原扫描规模与输入要求；执行前应检查它们是否符合实验目的。新研究优先使用明确传入输入、规模和成本来源的两个 CLI。
