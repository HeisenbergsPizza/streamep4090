# 从已有 Top-k trace 派生单分支路由

`simulation.project_routing` 是独立的离线输入模式。它不修改模拟器的路由回放、调度、计算成本或通信逻辑，也不修改原始 trace。生成的 Parquet 可直接交给现有 simulation 入口。

在 `StreamInfer` 目录中，使用已安装 `simulation/requirements.txt` 依赖的 Python：

```bash
python -m simulation.project_routing \
  --input reports/simulation/qwen16fields_20261005_4gpu_proxy/trace/routing.parquet \
  --output-dir /tmp/streamep-top1-seed20261020 \
  --mode uniform-from-topk \
  --seed 20261020 \
  --experts 128 \
  --source-manifest reports/simulation/qwen16fields_20261005_4gpu_proxy/trace/sample_manifest.json
```

输出目录必须不存在或为空，生成：

- `routing.parquet`：保留全部 `(rid, token_index, layer)` 键，仅有一个 `expert_logical_k0` 列。
- `projection_manifest.json`：记录源文件 SHA256、seed、原 K、新 K、每个原始列位置的抽取次数、每个专家的前后分配量，以及证据边界。

模式先按 `(rid, token_index, layer)` 排序，再用 `random.Random(seed).randrange(K)` 为每行选择一个原始专家。相同逻辑行集合与 seed 得到相同路由选择，不依赖原 Parquet 的物理行顺序。专家 ID 不重编号。原始专家列的位置不假定是 router score 排名。输入会经过现有 `ProfileDrivenRouter` 的完整性检查。

## 运行 decoding simulation

将生成的 `routing.parquet` 作为 `--routing-profile`，同时显式设置 `--top-k 1`。保持现有 Expert 成本曲线，按实际专家 batch 查表，不把成本除以原 K：

```bash
python -m simulation.steady_experiment \
  --routing-profile /tmp/streamep-top1-seed20261020/routing.parquet \
  --expert-profile simulation/expert_costs_profiles/Qwen3-30B-A3B_A100.csv \
  --attention-profile simulation/attention_costs_profiles/qwen3_30b_a3b_rtxpro6000_tp1/attention.csv \
  --output-dir /tmp/streamep-top1-defrag \
  --mode async --policy defrag --expert-scheduling layer \
  --gpus 4 --layers 48 --experts 128 --top-k 1 \
  --output-tokens 32 --initial-context 1024 \
  --active-request-cap 1024 --attention-batch-cap 256 --expert-batch-cap 1024 \
  --hidden-dim 2048 --bytes-per-element 2 \
  --network-bandwidth-gbps 800 --network-latency-us 5 \
  --lookahead-steps 4 --lookback-steps 4 --weight-decay 0.8 \
  --rate 200 --duration-s 5 --warmup-s 1 --block-s 1 \
  --trace-duration-ms 200 --trace-interval-ms 0.1 --seed 20261011

python -m simulation.plot_scheduler_trace \
  --traces /tmp/streamep-top1-defrag/scheduler_queue_trace.npz \
  --labels "Defrag / layer" --gpu 1 \
  --start-ms 4000 --end-ms 4200 \
  --output-dir /tmp/streamep-top1-defrag-figures \
  --title "Uniform single-branch routing from Top-8: simulation proxy"
```

`scheduler_execution.png` 展示所选 GPU 实际执行的 Attention/Expert service 区间与模型层；它不是某一条请求的路径。`scheduler_heatmap.png` 展示 ready 队列深度的时间均值。现有绘图入口要求一次调用中的 trace 具有相同 K，Top-1 和 Top-8 应分别绘制。

## 本次对照实验

`reports/simulation/qwen_top1_uniform_20261020/plan.json` 保存全部配置：K=1/8，各运行 Defrag/layer、FLFS/layer、MTFS/layer、FIFO/layer、Defrag/expert 和 Sync EP。到达 seed、请求映射、模型维度、成本和 batch 上限相同。Top-1 抽样 seed 为 20261020；到达 seed 为 20261011。实验记录 5 秒到达后完整排空；指标窗口为 1–5 秒，scheduler 图窗口为 4.0–4.2 秒。

```bash
python reports/simulation/qwen_top1_uniform_20261020/run_cases.py --workers 3
python reports/simulation/qwen_top1_uniform_20261020/analyze.py
python reports/simulation/qwen_top1_uniform_20261020/render.py
```

runner 会验证配置、输入和代码哈希后跳过已完成 case，拒绝覆盖不匹配或不完整的结果。

## 解释范围

派生数据属于固定原模型路由轨迹的反事实实验。删除某层分支后，后续层仍使用原 trace；没有重新执行 Top-1 模型前向。它不能代表原生 Top-1 模型的路由或生成质量。

当前数据组合仍是 Qwen3-235B 路由、Qwen3-30B/A100-labelled Expert GEMM 代理与 RTX PRO 6000 Attention profile。短窗口、单组 seed 的结果用于观察调度机制，未校准为真实硬件性能，也不估计稳定吞吐上限。Top-k 改变同时影响工作量、组批、通信和汇合依赖，无法把全部性能变化归因于汇合。
