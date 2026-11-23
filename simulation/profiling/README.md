# StreamInfer 单节点性能采集

本目录把目标节点的原始计时、设备配置和 simulator 可读取的成本表保存在同一个 profile 目录。当前支持单节点多 GPU，物理互联为 PCIe 或 NVLink，BF16、Attention DP=N/TP=1、Expert EP=N，专家均分。运行 simulator 时各卡共用一套计算成本表，因此需要同型号 GPU。默认模型形状为 Qwen3-30B-A3B；`init --model-config /path/to/config.json` 可读取兼容 MoE 的 HF 配置。算子使用 StreamInfer 合成推理路径和目标形状，执行单个 resident layer，无需下载权重。

## 在现有 4×4090 节点采集

在有已编译 StreamInfer CUDA 扩展、当前项目适配的 vLLM 和 PyTorch 的环境中，从仓库根目录运行。以下是该 PCIe/no-P2P 服务器的 Bash 示例。它复用了 benchmark 的 no-GDR、NCCL SHM 和 Flash Attention 设置。

```bash
cd "$HOME/guest0/projects/Qwen3/streamep4090"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate streamep4090
source experiments/scripts/rtx4090/common.sh
rtx4090_env
rtx4090_libraries
mkdir -p "$TMPDIR"

BUNDLE="/workspace/zzc/streamep4090/profiles/qwen30b-4090-graph-$(date +%Y%m%d-%H%M%S)"
python -m simulation.profiling init --bundle "$BUNDLE" --interconnect pcie --cuda-graph
python -m simulation.profiling collect --bundle "$BUNDLE" --expert-cap 512 \
  --attention-batches 1,8,16,32,64,128 --contexts 64,96,128,160,191
python -m simulation.profiling validate --bundle "$BUNDLE"
```

64 是第一步之前的 KV 历史长度。128 个 decode token 对应历史 64 到 191（含两端）。`collect` 默认在逻辑 GPU 0 测单卡算子，然后使用所有可见 GPU 测通信。各阶段在独立进程中顺序执行，避免上一阶段的 Graph 池占用下一阶段的显存。阶段成功后才发布；中断后可用 `collect --stages attention reformation network` 等指定剩余项。单独命令可以设置更细的采样范围。例如先看一条专家曲线时只执行 `python -m simulation.profiling expert --bundle "$BUNDLE" --max-batch-size 512`。已完成的阶段保留原始记录，再测同项可初始化新的目录。

在 NVLink 节点上，先进入该节点实际推理使用的环境，按部署设置 `CUDA_VISIBLE_DEVICES`，初始化时选 `--interconnect nvlink`。4090 的 `rtx4090_env` 会设置 `NCCL_P2P_DISABLE=1`，只在复现该 4090 配置时使用。清单会保留物理拓扑、CUDA peer access 矩阵和 NCCL 环境；NCCL 实际选择的通道以 `network/details.json` 及每个 rank 的 NCCL 日志为准。改变 Graph 设置时新建 profile，eager 模式在初始化时省略 `--cuda-graph`。操作系统、驱动和 package 版本也随清单保存。

## 每阶段的输出与边界

| 子命令 | 变化的量 | 可供模拟器使用的数据 |
| --- | --- | --- |
| `expert` | 单个专家接收的 BF16 行数从 1 到 cap | `expert/expert.csv`，单专家完整 forward 的毫秒数；覆盖所有整数行数 |
| `grouped` | 本地各专家的有序行数向量，包含零负载专家 | `grouped/grouped.csv`，当前后端下一整组的实测毫秒数。默认生成集中、部分活跃和均匀分布样本 |
| `attention` | 活跃请求数 × 每条请求已有 KV 历史 | `attention/attention.csv`，含 QKV、RoPE、KV、attention、输出投影及当前合成层的路由。覆盖完整二维网格；batch 内使用等长历史 |
| `reformation` | 输出行数、Top-K 输入行数、源片段数量及布局 | `raw.csv` 保存 CUPTI 单次 kernel 时间，`parameters.json` 写入可识别的普通 gather 与 gather-and-sum 参数；连续块直通时设备开销为零 |
| `network` | 消息字节数、每对卡的两个方向、并行与双向传输 | 各 rank 原始 CSV、NCCL 日志与单向接收端拟合。延迟单位 µs，带宽单位十进制 Gbit/s；并发场景保留用于判断争用 |
| `host` | 已有 Nsight Systems trace 中一个引擎线程的阶段、对应工作批和已知阻塞区间 | `host/raw.csv`、`details.json` 和可选有效 CPU 系数；使用方式见下节 |

对 expert/attention，每个格点先 warm up，再用 CUDA Events 计时，CSV 取中位数，`raw.jsonl` 保留各次毫秒数。Graph 模式使用原生 executor 的 capture bucket 和 replay，原始记录区分逻辑 batch 和执行 bucket。Events 测的是单次 native forward 在流上的时间跨度，eager 提交不及时产生的间隙也可能落在其中。做 CPU/GPU 分项分析时用 Nsight 对照实际 kernel 区间，避免把同一段间隙再次计入 CPU。计时使用重复输入和单层工作集，未模拟整模型跨层的缓存压力。

若有真实调度选批的本地专家行数，可用分布更贴近实验的向量替代默认 grouped 网格。初始化新目录后，把文件传给 `collect --vectors FILE` 或单独运行：

```bash
python -m simulation.profiling grouped --bundle "$BUNDLE" --max-batch-size 512 \
  --vectors "$VECTORS" --max-vectors 100
```

输入为 JSONL 或 JSONL.gz，每行一条本卡本批的整数数组，或带 `expert_counts` 的 Expert decision。4 卡 128 专家时每条向量恰有 32 个槽位。也可输入 simulator 的 `*.decisions.jsonl.gz`，此时覆盖率代表模拟调度样本。每次 forward 的 counts 来自一个实际被选中的本地批；原始 token 路由 trace 还需要根据放置与选批聚合。`details.json` 记录输入样本数、选中的向量数及命中次数。

## CPU 调度和已有 Nsight 报告

使用同一模型、Graph 设置、拓扑和负载下的真实推理报告。普通运行至少需要 engine 的 CUDA、NVTX 标记；启用 `DMOE_GAP_WORKER` 并保留 `WG.work`、`WG.retire` 时，可以更准确地把 postprocess 归到上一批的阶段。[采集示例](../../docs/rtx4090-profiling.md)

```bash
# REPORT 指向一个 worker 的 .nsys-rep
nsys export --type sqlite --output "$BUNDLE/worker.sqlite" "$REPORT"
python -m simulation.profiling host --bundle "$BUNDLE" --sqlite "$BUNDLE/worker.sqlite"
```

有多个引擎线程时传 `--tid GLOBAL_TID`。截取稳定窗口可用 `--start-ns A --end-ns B`，这里使用 SQLite trace 时间轴上的绝对纳秒。工具按同一进程的 CUDA correlation ID 匹配 gather launch 和 GPU kernel，分出提交前、launch API、提交后；1/64 采样的 scheduler 区间只在能关联到真实选批时用于估计。减去已知阻塞调用和 event 等待后，剩余墙钟仍可能包含线程被抢占、未识别的等待。默认仅输出观测。若在此次首次导入时加 `--use-wall-proxy`，会写入可归属的有效系数，供同类工作负载的敏感性实验使用。这些值带有采集和归属说明，不等同于实测的 CPU 活跃周期。

## 把 profile 交给 simulator

整个目录可复制到只有 CPU 的机器。工作负载路由另用与目标模型匹配的 Parquet 提供。示例沿用当前 512 并发上限、每条请求输入 64、输出 128 的设定：

```bash
python -m simulation.steady_experiment --profile-bundle "$BUNDLE" \
  --routing-profile "$ROUTING" --mode async --policy defrag \
  --active-request-cap 512 --attention-batch-cap 128 --expert-batch-cap 512 \
  --initial-context 64 --output-tokens 128 --rate 12 \
  --duration-s 40 --warmup-s 10 --block-s 5 --output-dir "$BUNDLE/steady-r12"
```

`simulation.run` 也接受 `--profile-bundle`。两种入口自动读取设备数、模型维度、dense expert 表、可用的 Attention 表和适用参数，async 默认使用 `pipeline`。显式缩小 cap 或修改拟合参数可以用于消融；维度冲突、缺少 Attention 网格端点或超出 expert 表范围时会报错。Grouped 表须显式加 `--use-bundle-grouped`，模拟时遇到未测向量会报错；默认使用已有的单专家求和模型。`validate` 会列出已有阶段、覆盖范围、已拟合参数和仍没有估计的参数。仅有原始样本或阶段目录不代表拟合成功，例如 PCIe 拓扑中的异质链路可能超出单一带宽参数所能描述的范围。

部署配置及单位由 `manifest.json` 和各阶段的 `details.json` 保留。模拟器导入时验证表格和参数文件的哈希，可在模拟输出中追踪实际载入的参数和仍使用默认值的部分。输出中沿用的 `hardware_calibrated=false` 表示整机性能尚待独立验证。评估可另外保留一档 arrival rate，比较实机与模拟的 ITL、吞吐、批大小和 GPU 空档。

采集命令帮助：`python -m simulation.profiling <子命令> --help`。CPU 端回归测试从仓库根目录执行 `python -m unittest discover -s simulation/profiling/tests`。
