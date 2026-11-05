# 四张 RTX 4090 上的 CPU / GPU profiling

先用 **Nsight Systems** 看线程、CUDA API、GPU kernels、内存复制及它们的重叠；确定慢在哪个 kernel 后，再考虑用 Nsight Compute 单独分析该 kernel。当前入口为 `experiments/scripts/rtx4090/profile.sh`。

## 测量范围

这是本仓库的实际 GPU 执行路径：四个 Ray Engine、CPU 调度和通信线程、attention、grouped GEMM、dispatch、Top-k 归并等。权重、hidden states 和路由仍是合成数据，没有加载 Qwen checkpoint，也不包含真实 prompt prefill。结果可分析当前实现与硬件上的开销；真实 Qwen 请求的路由、批次分布和文本生成性能还需另行验证。

每个 Engine 单独生成一份报告，覆盖其 Python 和原生线程。driver、tokenizer、detokenizer 进程的 CPU 活动不在这四份报告中；完整请求排队时间和每个 token 的 Top-k 等待也不能仅靠这次 kernel trace 还原。

本地已做 CPU 单元测试、脚本检查和 dry-run；没有 CUDA/Nsight 环境，尚未在服务器验证报告内容。

## 1. 安装 Nsight Systems CLI

在现有 Ubuntu 22.04 x86_64 Docker 容器内安装 CLI，继续使用已经跑通 benchmark 的 Conda、CUDA Toolkit 和驱动。下面固定为 NVIDIA 官方仓库提供的 2025.6.1，约 197 MB；可从[官方包目录](https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/)核对。

```bash
mkdir -p "$HOME/downloads/streamep"
cd "$HOME/downloads/streamep"
curl -fL --retry 3 \
  -o NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb \
  https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb
sudo apt-get install -y ./NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb
nsys --version
nsys status --environment
```

`status` 若提示 CPU sampling/context switch 权限不足，先运行下文的 `trace` 模式：它显式关闭这两项，仍采 CUDA、NVTX 和部分 OS runtime 库调用。`cpu` 模式取决于宿主机 perf 限制及 Docker seccomp/capability 配置。不要因 CPU sampling 检查失败就重装驱动或以 sudo 运行整个 Ray benchmark。需要时把检查结果交给容器管理员，按 [NVIDIA CPU profiling 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cpu-profiling-on-linux)处理。

先在空闲 GPU 上做一个短检查，验证 CLI 能采到 CUDA activity：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate streamep4090
tool_check_dir="$(mktemp -d /tmp/streamep-nsys-check-XXXXXX)"
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=none \
  --output="$tool_check_dir/cuda" \
  python - <<'PY'
import torch
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
torch.cuda.profiler.start()
with torch.cuda.nvtx.range("tool_check.matmul"):
    for _ in range(10):
        y = x @ x
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("Profiler stop returned; Python continued successfully.")
PY
nsys stats "$tool_check_dir/cuda.nsys-rep"
```

工具检查中的 synchronize 用于完成独立测试。正式 benchmark 没有在每个阶段新增同步。

## 2. 上传修改并重编本仓库扩展

上传当前 StreamInfer 源码，保留服务器已有依赖和已初始化的 `third_party` 目录，排除本地编译的 `.so` 和 `build/`。本次既有 Python 修改，也有 C++ NVTX 桥接，单独上传 `profile.sh` 不够。

下面的重编步骤用于首次安装带 NVTX 桥接的 profiling 版本。如果服务器已经成功生成过这套入口的四份报告，本次新增的 `--synchronized-start` 只修改 Python，无需因该功能重新编译 C++ 扩展；同步源码后可直接运行第 3.1 节。

在服务器 **StreamInfer 根目录**执行以下命令。Conda 环境应保持原来可以编译和运行 benchmark 的 CUDA 路径设置，参见 [环境配置](rtx4090.md)。

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate streamep4090
export PYTHON=python
source experiments/scripts/rtx4090/common.sh
rtx4090_env
rtx4090_libraries
mkdir -p "$TMPDIR"
export MAX_JOBS=4
test -f third_party/NVTX/c/include/nvtx3/nvToolsExt.h
"$PYTHON" -m pip install -e . --no-build-isolation --no-deps

DMOE_ENABLE_NVTX=1 ENABLE_NVTX=0 "$PYTHON" - <<'PY'
import torch
import disagmoe_c as c
assert c.nvtx_enabled(), "The updated native NVTX bridge is not enabled"
print("Native NVTX bridge is enabled:", c.__file__)
PY
```

`nvtx_enabled()` 只检查扩展及开关，采集成功仍需查看实际报告。它通过独立的 `DMOE_ENABLE_NVTX=1` 输出 NVIDIA NVTX，保留旧内存 recorder 的 `ENABLE_NVTX=0`。

## 3. 先采一对短实验

```bash
# 查看配置：不导入 CUDA、不启动 Ray、不生成文件，也不要求安装 nsys。
bash experiments/scripts/rtx4090/profile.sh trace --dry-run

# 同样的负载，先记录无 profiler 的性能。
bash experiments/scripts/rtx4090/profile.sh baseline

# warmup 后开启采集，所有 benchmark 请求完成后关闭。
bash experiments/scripts/rtx4090/profile.sh trace
```

三个模式默认均为完整 48 层、8 个请求、1 request/s、初始上下文 64、输出 32、attention cap 16、expert cap 128、BF16、CUDA Graph 关闭。沿用 benchmark 的 warmup：10 个请求、初始上下文 100、输出 50。第一轮用于核实采集和观察低负载行为，不能作为节点的饱和吞吐测试。

需要更多样本时，对 baseline 和 trace 使用相同参数，例如：

```bash
bash experiments/scripts/rtx4090/profile.sh baseline --num-requests 32 --rate 2
bash experiments/scripts/rtx4090/profile.sh trace --num-requests 32 --rate 2
```

`--rate` 是请求到达率，batch cap 是上限；二者都不等于 GPU 上实际形成的 batch。比较时固定层数、长度、路由方式、scheduler、graph 设置，记录实际 batch 分布。提高到达率前先检查显存与是否出现持续积压。

`nsys status --environment` 确认支持 CPU profiling 后，单独补采：

```bash
bash experiments/scripts/rtx4090/profile.sh cpu
```

该模式增加 process-tree CPU sampling 和 context switches，观察 Engine 的 CPU 热点及线程运行/等待。调用栈能否展开到 Python 函数还取决于采样与符号支持；NVTX 可直接给关键 Python 阶段命名。短窗口的热点比例可能波动。

不要在 `profile.sh` 外再套一层 `nsys profile`，也不要同时传 `--profile-dir`、`--trace` 或 advanced logging。Ray 已为每个 GPU Engine 启动 Nsight；两个 CUPTI collector 可能互相冲突，额外日志也会影响 CPU 开销。

### 3.1 统一起跑，再观察异步推进

使用可选的 `--synchronized-start`，让 64 个请求在四卡上各形成一个 **B=16 的首层 attention batch**。默认入口保持原有行为；单独设置 `--generator-type offline` 仍然只是集中提交请求。

```bash
# 先检查参数，无需 GPU 或 Nsight。
bash experiments/scripts/rtx4090/profile.sh trace --dry-run \
  --synchronized-start --generator-type offline --num-requests 64

# 同样的起跑条件，先采无 profiler 的基线。
bash experiments/scripts/rtx4090/profile.sh baseline \
  --synchronized-start --generator-type offline --num-requests 64

# 再采四个 Engine 的 CPU / CUDA 时间线。
bash experiments/scripts/rtx4090/profile.sh trace \
  --synchronized-start --generator-type offline --num-requests 64
```

这组命令沿用初始上下文 64、输出 32、attention cap 16、expert cap 128 和 48 层设置。`offline` 下 `--rate` 不控制起跑时刻。长度参数按 `[min,max)` 生成，因此固定长度 64 对应 `--min-input-len 64 --max-input-len 65`。

起跑流程：

1. 完成原有 warmup，检查四卡都能容纳本轮分配的 KV blocks。
2. 固定分配 16 个请求到每个 DP rank；各推理线程准备完整的首层 attention batch，等待本卡准备工作完成。
3. 等全部 Engine 就绪，再开启所有 profiler，发布同一个未来 CPU 时刻。
4. 各线程在该时刻附近提交首层计算，此后继续原有异步调度，不再设置层间或 token 间的全局起跑门槛。

后续 attention / expert batch 大小由就绪队列决定，可能小于或大于初始 B=16（受各自 cap 限制）；每张卡也可能交错处理不同层。一个 token 仍要等自己的 Top-8 专家结果齐全，才能推进下一层。

目前该模式要求单节点四卡、colocate、TP=1、DP=EP=4、offline、固定输入和输出长度；请求总数必须能被四整除，每卡初始请求数不能超过 attention cap。所有初始请求的 KV blocks 必须同时放得下。`server` 和 `--analyze-throughput` 暂不支持。

**统一的是准备条件和 CPU 放行时刻。** Python 线程、OS 调度、CUDA 提交及 GPU 排队仍会产生起跑偏差，不能据此声称四卡 kernel 严格同时开始。每轮会额外写出 `metrics.csv.synchronized-start.json`：检查四个 worker 的 `initial_batch_size` 都为 16、`layer_id` 都为 0、`req_ids` 覆盖全部 64 个请求，再查看 `cpu_start_skew_ms`。它是各线程 `execute_begin_ns` 的最大差值，实际 GPU 起跑偏差还需看 Nsight 中关联的第一批 kernels。

trace 中可搜索：

- `Engine.synchronized_start.wait`：等待统一放行的 CPU 区间；该范围从采集前开始，报告中可能截断。
- `Engine.execute.attn.layer=0.batch=16.initial`：每卡首批执行提交。
- `Engine.execute.attn.layer=L.batch=B` / `Engine.execute.expert.layer=L.batch=B`：后续异步执行提交；层号从 0 开始。

首批准备在采集开始前完成，等待统一放行的区间在 trace 内；benchmark 总时长从共同放行时刻开始计。比较异步分散时，保留各报告的共同时间基准，并先量化首批实际 GPU 起跑偏差。不要把每份报告的首个 kernel 各自平移到零后，再用它们判断跨卡通信重叠。

该模式仍执行合成 decode，使用随机权重和 hidden states；上述命令默认使用随机路由，已有路由配置仍生效。未增加真实 Qwen checkpoint 或 prompt prefill。

## 4. 输出与查看

默认每次写入新的 `reports/rtx4090/profile-<mode>-<time>-<pid>/`：

- `metrics.csv`：本轮 benchmark 指标。
- `metrics.csv.synchronized-start.json`：仅统一起跑模式生成，包含初始分配、准备/放行/提交时间及 CPU 起跑偏差；baseline 也会生成。
- `run.log`：运行日志。
- `nsys-version.txt` / `nsys-environment.txt`：工具与环境检查，baseline 无此项。
- `nsight/worker_process_<pid>.nsys-rep`：四个 Engine 的报告。
- `nsight/manifest.json`：配置、Nsight 参数、Ray 目录、Engine PID / device_id / CUDA_VISIBLE_DEVICES 映射及测量范围；统一起跑模式还包含起跑报告。

可设置 `PROFILE_RUN_DIR=/absolute/path/to/new-run`，目录必须不存在。Ray actor 通常只看到自己的 `cuda:0`；判断物理卡时结合 manifest，不能把四份报告的 cuda:0 当成同一张卡。

日志中的 `PASS: benchmark completed` 只表示 benchmark 完成。随后应有 `PROFILE: copied 4 reports ...`，再验证报告可读且包含 CUDA kernels 和自定义 NVTX：

```bash
# 替换为本轮实际目录。
profile_run_dir="reports/rtx4090/profile-trace-REPLACE_WITH_ACTUAL_RUN"
for report in "$profile_run_dir"/nsight/*.nsys-rep; do
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum \
    "$report" > "${report%.nsys-rep}.stats.txt"
done
```

检查四份 stats 是否出现 kernel 表及 `Engine.*`、`MoE.CUTLASS.*` 等范围。`stats` 会生成 SQLite 中间文件，占用额外磁盘空间；某个类别没有活动可能被跳过，不能仅凭退出码判断 CUDA/NVTX 数据齐全。

时间线需要 Nsight Systems GUI：把 `.nsys-rep` 下载到安装了兼容 GUI 的机器打开，优先使用同版或更新版。按[安装指南](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)确认本机系统/架构支持，服务器只需 CLI。统计表适合找累计热点，重叠和关键路径要看 timeline。

采集使用 `capture-range=cudaProfilerApi`、`capture-range-end=stop-shutdown`、`kill=none`：结束采集 session 后让 Engine 继续完成指标与清理；`wait=primary` 要求 CLI 等待 actor 退出。四个 worker 通过 RPC 分别开关，边界有少量时间差。窗口包含请求到达间隔和尾部 drain，不能直接称为稳态。CLI 语义见 [Nsight 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options)。

包装器在关闭自己的 Ray 前，最多等 60 秒收齐非空、连续两次检查大小/mtime 不变的报告，然后复制；原文件保留在打印的 `/tmp/streamep-ray-*/session_*/logs/nsight/`。稳定检查只能粗略判断落盘，不能替代解析。超时应保留该目录和 worker/runtime_env 日志，核对采集选项、工具错误与导出状态。现有 launcher 的 900 秒整程 timeout 包含初始化和导出，长实验可设置 `RUN_TIMEOUT_SECONDS`。

## 5. 各项开销怎么看

| 项目 | 主要证据 | 解读限制 |
| --- | --- | --- |
| CPU 调度调用 | `Engine.schedule.sample_1_in_64` | 每 64 次轮询采样，含空轮询；不能直接累加为全部调度成本 |
| 非空队列出队、batch 准备与 merge 提交 | `Scheduler::dequeue_merge_unified` | 含 CPU 工作和异步 CUDA 提交，GPU 完成另看 CUDA activity |
| CPU 预处理、执行提交、后处理 | `Engine.preprocess` / `execute` / `postprocess` | NVTX 是墙钟区间，可含等待、抢占和嵌套调用 |
| 等前一个 batch 的事件 | `Engine.previous_event_wait` / `Engine.stream_sync` | 结合 GPU timeline 找依赖未就绪的原因 |
| attention、norm、投影、routing | CUDA kernels、已有 attention 范围及 CUDA API 关联 | GPU 执行与 CPU 发起调用分别统计 |
| grouped expert GEMM、激活 | `MoE.CUTLASS.up` / `silu_gate` / `down` 及关联 kernels | Python 范围含 metadata 准备/launch，矩阵运算看 GPU 行 |
| token 重排、Top-k gather/sum | permute / gather / sum kernels、出队范围、H2D copies | CPU copy、指针表 H2D 和 GPU gather 是不同部分 |
| 通信提交与传输 | `MuDispatcher::*`、NCCL kernels、memcpy、通信线程 | NCCL kernel 可含等待；SHM/host 路径不能概括为直接 GPU P2P 带宽 |
| CPU 热点、线程运行/阻塞 | `cpu` 模式 samples、context switches，加 OSRT | 需要权限；OSRT 覆盖部分 runtime 函数，不是全部系统调用 |

分析顺序：

1. 对比 baseline 与 trace 的吞吐/ITL，评估 profiler 扰动。不要同时修改日志级别或 scheduler。
2. 每卡检查 GPU 空档：请求少、CPU 尚未提交、等待其他专家，还是通信依赖？结合相邻 CUDA API、NVTX 和线程活动定位。
3. 从 kernel 统计找到 attention、GEMM、reformation、通信的累计热点，再在 timeline 看是否重叠。
4. 分别报告调用次数、均值/分位数、资源活动时间、暴露在请求关键路径上的等待。多 stream 的 kernel activity time 应取时间区间并集；它表示存在运行中 kernel 的时间，NCCL kernel 还可能在等待，不能直接当成 SM 利用率。嵌套 CPU 范围区分 inclusive/exclusive。

**不能把 CPU 时间、各 GPU kernel 累计时间、通信时间直接相加当成 ITL。** 不同线程、stream 和 GPU 之间存在重叠；四卡累计 kernel 时间也可能超过整段实验墙钟时间。

若开启 CUDA Graph，Python up/gate/down 范围不会随 graph replay 重新执行；当前配置使用 `cuda-graph-trace=node` 展示内部 kernels，采集开销更高。先完成 eager 基线，再单独比较 graph 配置。

`TOPK_STATS age` 是 CPU 处理首个专家返回到处理第 K 个返回的间隔，含分支返回差异、通信和接收处理等，不能称为 gather kernel 开销。该统计跨 layer 汇总，不能直接当成每层等待分布。原 advanced logging 的 `queuing_delay` 来自 schedule 调用耗时除以 token 数，不能当作 token 入队到出队的等待。精确拆每 token 的 queue wait / Top-k join wait，需要带 token、layer、入队/到齐/出队时间的事件记录，并控制记录开销。

原 `--profile-dir` 使用 PyTorch profiler，默认启动条件要求 batch 至少 100；四卡入口 attention cap 16 时可能没有期望的 attention 文件。旧 `--trace` 在当前 overlap/no-GDR 路径也不覆盖所有阶段，因此第一轮采用上述 Nsight 路线。
