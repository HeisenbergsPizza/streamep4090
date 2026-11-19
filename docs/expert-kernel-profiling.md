# 单卡 Expert 算子对照与 Nsight Compute 采集

## 测单个 expert 的 batch–耗时曲线

在已能运行 StreamInfer 的 CUDA 环境中，从仓库根目录运行下面的命令。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/ops/sweep_expert_batch.py
```

横轴是**送入同一层、同一个 expert 的 token 行数**。默认使用 Qwen3-30B-A3B 的形状（hidden 2048、expert intermediate 768）及随机 BF16 权重。脚本复用该 expert，预热每个形状，再用 CUDA Event 记录 20 次 eager forward 的中位耗时。计时包含 gate/up 投影、SiLU 门控乘法和 down 投影，CUTLASS 路径还包含两次 metadata kernel。路由、跨卡通信和输入准备都在计时范围外。

脚本根据所选 GPU 启动时的空闲显存预留 20%，扣除权重并估算各行激活占用，依次测量 1、2、4、8 等 batch 和估算终点。遇到 CUDA OOM 会停止增长，保存此前成功的点。终端打印最大实测 batch，CSV 和 PNG 默认保存在 `reports/expert-batch/<时间戳>/`。

只看曲线可以加 `--max-batch-size 1024`。要给模拟器准备查找表，下面的命令会逐个测量 1 到 512 范围内显存允许的 batch。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/ops/sweep_expert_batch.py \
  --dense --max-batch-size 512 --output-dir reports/expert-qwen-b512
```

测满 1 至 512 后，可将该目录的 `expert_latency.csv` 传给模拟器的 `--expert-profile`，用于 expert cap 不超过 512 的实验。默认倍增扫描只覆盖抽样点。这里的 `time_ms` 是单个 expert 完整前向的实测值；使用该表时，模拟器仍按本地各 expert 的负载分别查表并求和。

## 对照 StreamInfer 和 SGLang 的 kernel

这组脚本比较 StreamInfer 的 BF16 CUTLASS expert 与 SGLang **0.5.9** 的 Triton expert。两边读取同一个测试文件，复用相同输入、语义等价的权重、路由和路由权重。它运行一层、一个 EP rank 的本地计算，不启动服务、Ray、NCCL 或 Nsight Systems，也不读取完整模型权重。

新增入口：

- `experiments/scripts/ops/expert_bench.py`：生成测试输入、检查结果、CUDA Event 计时、标记采集范围。
- `experiments/scripts/ops/sglang_expert_adapter.py`：对接 SGLang 0.5.9 的原生 Triton MoE，给两次 GEMM 和激活加采集标记。
- `experiments/scripts/ops/profile_experts.sh`：先正常验证和计时，再启动一次 Nsight Compute 采集。

这是一组受控微基准。它可以定位算子和数据组织成本，不能直接代表四卡推理的端到端性能。服务器上的 GPU 计算、SGLang 导入和 NCU 兼容性仍需按下面的步骤验证；本地 CPU 检查不替代这些验证。

## 1. 同步文件与选择 GPU

把上述三个脚本同步到服务器当前仓库，目录结构保持一致。下面按已有路径编写，所有命令均在**服务器终端**执行：

```bash
cd /home/guest/guest0/projects/Qwen3/streamep4090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate streamep4090

export EXPERT_WORK=/workspace/zzc/expert-compare
export EXPERT_REPORT_ROOT="$EXPERT_WORK/reports"
export NCU_BIN=/home/guest/miniforge3/envs/streamep4090/bin/ncu
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$EXPERT_WORK/cases"

nvidia-smi
"$NCU_BIN" --version
```

确认选择的 GPU 没有其他训练、推理或 profiling 工作。需要换卡时修改 `CUDA_VISIBLE_DEVICES=0`；脚本只接受一个可见设备，程序内部使用 `cuda:0`。不要同时在两套环境里采同一张卡。

`NCU_BIN` 指向现有 2024.3.2 的绝对路径。之后切换 SGLang 环境，仍可用这个工具启动该环境的 Python，无需把 SGLang 装进 `streamep4090`。`nvidia-smi` 显示的 CUDA 13.2 是驱动能力，不能保证这版 NCU 支持两个环境实际使用的所有 CUDA API。先进行下面的小范围采集；若报告 CUDA/API 不支持，再核对该环境的 `torch.version.cuda` 和 NCU 支持情况。

## 2. 生成同一份输入

先在 `streamep4090` 环境运行一次：

```bash
python experiments/scripts/ops/expert_bench.py prepare \
  --output "$EXPERT_WORK/cases/balanced-local512.pt" \
  --tokens 256 --routing balanced --seed 0
```

默认模型形状与 Qwen3-30B-A3B 一致：hidden 2048、intermediate 768、128 个全局专家、Top-8、EP4，选择 rank 0 的 32 个本地专家。均衡路由下：

| `--tokens` | 全局 token 数 | 本地 expert 分支数 | 每个本地专家的行数 |
|---:|---:|---:|---:|
| 256 | 256 | 512 | 16 |
| 512 | 512 | 1024 | 32 |

这里 `--tokens` **不是 expert cap**。第一组用于对齐一个满 cap512 的本地 expert 调用。第二组表示一次处理 1024 条本地分支，不能直接视为原运行时 cap512 的行为；原运行时需要将它们分批。随机路由下本地分支数和各专家行数会波动，应以结果元数据为准。

相同 seed 跨两个环境重新生成数据，不能充分保证输入一致；这里直接共享同一个 `.pt` 文件。只使用自己生成且信任的测试文件。

## 3. 先单独采 gate/up GEMM

### StreamInfer

仍在 `streamep4090` 环境、仓库根目录执行：

```bash
bash experiments/scripts/ops/profile_experts.sh streaminfer up \
  --case "$EXPERT_WORK/cases/balanced-local512.pt"
```

脚本会顺序执行两次 Python：

1. 普通运行：检查数值结果，预热，用 CUDA Event 测本框架 native forward 的延迟。
2. NCU 运行：单独预热，再采一次 `ExpertBench.up` 范围，保存 `.ncu-rep`。

StreamInfer 的 `up` 包含 `setup_meta` 和 grouped GEMM，报告里要分别看这两个 kernel。脚本没有设置 `--launch-count 1`，避免只采到 metadata。

若只想先检查命令和路径：

```bash
bash experiments/scripts/ops/profile_experts.sh streaminfer up \
  --case "$EXPERT_WORK/cases/balanced-local512.pt" --dry-run
```

`--dry-run` 不调用 GPU、Python 或 NCU，不创建报告目录。

### SGLang

等上一条命令退出后，切换到已有 SGLang 环境：

```bash
source /workspace/zzc/sglang-qwen30b/activate.sh
cd /home/guest/guest0/projects/Qwen3/streamep4090

export EXPERT_WORK=/workspace/zzc/expert-compare
export EXPERT_REPORT_ROOT="$EXPERT_WORK/reports"
export NCU_BIN=/home/guest/miniforge3/envs/streamep4090/bin/ncu
export CUDA_VISIBLE_DEVICES=0

python -c 'import importlib.metadata, torch; print("sglang", importlib.metadata.version("sglang")); print("torch", torch.__version__, "CUDA", torch.version.cuda)'

bash experiments/scripts/ops/profile_experts.sh sglang up \
  --case "$EXPERT_WORK/cases/balanced-local512.pt"
```

适配器面向 SGLang 0.5.9 的内部 API，并检查版本。版本不一致时应先调整适配器，不要为了通过检查修改已部署推理环境的版本。

## 4. 分别采 down、激活和完整本地流程

在各自环境中将上面的第二个位置参数 `up` 换为：

| 参数 | NCU 采集边界 | 比较重点 |
|---|---|---|
| `up` | gate/up 投影；StreamInfer 包含 metadata | 两边 GEMM 本身的时间、tile 与访存 |
| `activation` | SiLU 与门控乘法 | StreamInfer 的两个操作与 SGLang 的融合操作 |
| `down` | down 投影；StreamInfer 包含 metadata | GEMM、路由权重应用所处的位置 |
| `full` | 本框架的 native expert forward | 拆解本框架内的 kernel；两边边界不同，不能直接给框架性能排名 |

例如采 StreamInfer 的 down：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate streamep4090

bash experiments/scripts/ops/profile_experts.sh streaminfer down \
  --case "$EXPERT_WORK/cases/balanced-local512.pt" --detailed
```

再在 SGLang 环境采同一个阶段：

```bash
source /workspace/zzc/sglang-qwen30b/activate.sh

bash experiments/scripts/ops/profile_experts.sh sglang down \
  --case "$EXPERT_WORK/cases/balanced-local512.pt" --detailed
```

默认采集 `SpeedOfLight`、`LaunchStats`、`Occupancy`。`--detailed` 额外采集 `ComputeWorkloadAnalysis` 和 `MemoryWorkloadAnalysis`，通常需要更多 replay passes。先完成一组默认采集，再对有差异的 GEMM 使用详细模式。

单独采某个阶段时，程序依然计算 native forward 来提供正确的前置数据，NCU 只记录所选 NVTX 范围内的 kernel。初始化、参考结果和预热都在标记范围之外。两边都关闭 CUDA Graph。

采集边界保留当前实现：StreamInfer 是按专家排好序的输入经过 metadata、两次 GEMM 和激活；输入分组与为正确性验证所做的加权合并放在采集范围外。SGLang 原生 `fused_experts` 自带路由对齐、权重应用和局部合并。因此 `full` 和普通 CUDA Event 的两个总时间边界不同，**不能直接相除作为两种框架或完整 MoE 的加速比**。这里优先比较 up/down/activation 的实际 kernel，并单列 metadata、对齐、combine 等成本。

验证将两边结果整理为相同的 rank 0 本地 expert 贡献，尚未经过跨 EP rank 的求和。实验不包含通信和完整 Transformer。

不要在现有 `profile_server*.sh` 或 `profile_graph_b512.sh` 外层再套这个脚本：那些入口会启动服务、通信和 Nsight Systems，这里的入口专用于单卡采集。

## 5. 结果在哪里、怎么读

每次运行创建独立目录，例如：

```text
/workspace/zzc/expert-compare/reports/streaminfer-up-日期-时间-PID/
  timing.json          普通运行的验证结果、环境信息和 native forward 的 CUDA Event 时间
  capture.json         NCU 运行的元数据
  kernels.ncu-rep      用 Nsight Compute GUI 打开
  details.txt          NCU 指标的文本导出
  timing.log
  capture.log
  commands.txt
  ncu-version.txt
  nvidia-smi-before.txt
```

即使采集的是 `up`，`timing.json` 中的 `native_forward_ms` 仍表示**本框架 native forward**，具体边界见其中的 `backend_metadata`。单个 GEMM 的硬件指标和 NCU duration 在 `.ncu-rep`/`details.txt` 中。不要把 NCU 整个命令的运行时间当作算子延迟。

先检查输入文件、各专家行数、dtype、GPU、CUDA/PyTorch/Triton 版本及 SGLang 配置选择，再比较同一阶段。两套环境的软件版本也属于剩余变量，需要保留在报告里。

建议依次看：

1. 同样有效计算量下 GEMM duration 是否仍明显不同。
2. `ComputeWorkloadAnalysis` 中 Tensor Core 管线表现，以及计算吞吐占比。
3. `MemoryWorkloadAnalysis` 中 DRAM 流量、吞吐与 L2 命中情况。
4. grid/block、寄存器和共享内存占用，以及 occupancy 限制因素。
5. 辅助 kernel 数量及成本。NCU 会串行化和重放被采 kernel，Python 提交间隙仍应结合 Nsight Systems 分析。

脚本显式使用 `--clock-control none`，不更改共享 GPU 的时钟；频率、温度和其他任务仍可能影响结果。`--cache-control all` 在 replay 前清理 GPU cache，便于比较受控的硬件指标；它不等于真实推理中的热缓存状态。普通 CUDA Event 时间使用自然缓存行为，不能把两类测量差异全部归因于算子。有关 NVTX 过滤、replay、缓存和时钟，见 [NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/#nvtx-filtering) 和 [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#workload-durations)。

## 6. 常见失败

- **`ERR_NVGPUCTRPERM`**：当前容器用户没有读取 GPU 性能计数器的权限。普通推理和 Nsight Systems 成功不保证有该权限。把完整错误交给管理员核对宿主机设置和容器权限；在共享服务器上不要自行卸载驱动或重启。参考 [NVIDIA 权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)。
- **没有生成 `.ncu-rep` / `No kernels were profiled`**：检查 `capture.log`，确认使用的是新脚本，NVTX range 为 `ExpertBench.up` 等，NCU 过滤参数末尾有 `/`；它表示 push/pop 范围。
- **CUDA/API 不受支持**：保留 `ncu-version.txt`、`capture.log` 与 Python 环境版本，核对工具兼容性。需要更新时使用独立 NCU 安装并调整 `NCU_BIN`，无需更换已验证的框架环境。
- **数值检查失败**：先分析验证输出，不要跳过验证继续比较性能。两种布局、Top-k 权重和非本地专家屏蔽都必须语义一致。
- **报告目录已存在**：默认会生成新名字；手动指定 `--output-dir` 时也必须使用新目录，脚本不会覆盖旧报告。
