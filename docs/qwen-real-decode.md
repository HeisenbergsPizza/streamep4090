# 四张 RTX 4090 上运行 Qwen3-30B-A3B 真实 decode

实验用同一份本地 BF16 checkpoint。Transformers 计算一次 prompt prefill，导出每层 KV、首个输出 token 和一段 greedy 参考序列。StreamInfer 用自己的 Attention、CUTLASS expert 和异步调度执行后续 decode。当前入口支持单节点四卡、Attention DP4 / EP4 / TP1、等长 prompt、固定步数和 eager 执行。

从旧 checkout 更新时需要重新编译 `disagmoe_c` 才能使用多请求 top-k 分发修复。旧版扩展将不同请求的 ID 与专家输入行错配，尤其容易在使用不同 prompt 的混合 batch 时暴露。可以复用原来的 prefill 快照。在已配置好的 `streamep4090` 环境和仓库根目录执行：

```bash
export PYTHON=python
source experiments/scripts/rtx4090/common.sh
rtx4090_env
rtx4090_libraries
mkdir -p "$TMPDIR"
MAX_JOBS=4 python -m pip install -e . --no-build-isolation --no-deps
python -m pytest -q tests/ops/test_token_routing.py -k topk_metadata_and_hidden_rows_stay_together
```

## 1. 导出一个小样本

在服务器仓库根目录运行。使用支持 Qwen3 MoE 的参考环境，需有 CUDA PyTorch、Transformers、Accelerate 和 safetensors。导出器只读本地模型。以下命令用同一个 prompt 准备四个独立请求，方便检查不同 GPU 得到的结果。

```bash
cd "$HOME/guest0/projects/Qwen3/streamep4090"
source /workspace/zzc/sglang-qwen30b/activate.sh
python -c 'from transformers import Qwen3MoeForCausalLM; import accelerate, safetensors, torch; print(torch.cuda.device_count())'

export MODEL=/workspace/zzc/sglang-qwen30b/Qwen3-30B-A3B
export RUN_DIR="/workspace/zzc/streamep4090/real-decode/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/prompts.jsonl" <<'JSON'
{"text":"The capital of France is"}
JSON

CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/scripts/rtx4090/prepare_qwen_decode.py \
  --model-dir "$MODEL" --prompts-jsonl "$RUN_DIR/prompts.jsonl" \
  --output "$RUN_DIR/snapshot" --repeat 4 --reference-steps 4
```

每行也可直接写 `{"input_ids":[...]}`，可保留与 SGLang 相同的分词和 chat template。文本输入使用原文分词。导出器依次处理各行，`--repeat` 共享磁盘上的只读文件，每个运行中的请求有独立 KV 页。校验时请使用导出快照时的原始权重文件。快照记录配置和权重索引的 SHA256。

## 2. 在 StreamInfer 环境运行

继续使用同一终端中的 `MODEL` 和 `RUN_DIR`。启动器读取快照，自动设置请求数、输入长度、decode 步数、统一起跑和 defrag 调度。默认每个请求执行的步数为所有请求共同具备的参考步数。

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate streamep4090
export DMOE_RAY_TMPDIR=/workspace/zzc/ray
export RUN_TIMEOUT_SECONDS=1800
bash experiments/scripts/rtx4090/launch.sh benchmark \
  --checkpoint-path "$MODEL" --decode-snapshot "$RUN_DIR/snapshot" \
  --gpu-usage 0.80 --file "$RUN_DIR/metrics.csv"
```

同一命令加 `--dry-run` 可预检参数、快照清单和文件是否存在。实际运行会执行算子预热，再用这批 prompt 预热最多两步，释放请求并从快照恢复本轮的初始状态。正式计时从四卡共同放行到最后一个响应送达。每次 forward 包含 embedding、48 层计算、final norm、LM head 及 greedy 反馈。当前使用固定输出长度，包括模型选中 EOS 后的剩余步数。

- `metrics.csv.decode.json` 包含每个请求的 `prefill_token`、本轮实际输出 `token_ids`、参考前缀和首个差异的位置（从零开始）。若参考前缀有差异，程序保留该文件并报告失败，便于核对 BF16 数值偏差或实现错误。
- `metrics.csv` 在参考前缀匹配后记录 ITL、吞吐和整批耗时。`metrics.csv.synchronized-start.json` 记录请求与快照的对应关系、运行路径及各 GPU 起跑偏差。

首个输出 `y0` 已由 prefill 选出。快照有 64 个 prompt token 时，StreamInfer 在位置 64 输入 `y0`，写入该位置的 KV，并输出 `y1`。四次实测前向贡献四个输出（`y1…y4`）。prefill、权重加载、KV 导入和预热都在计时窗口外。ITL 从本轮实际输出的相邻 token 时间计算。

## 3. 扩大到原来的负载

用 512 行**各自恰好 64 个 token** 的真实输入重新导出，设 `--repeat 1`，输出到新目录。若已有 SGLang 实验输入，直接复用对应的 `input_ids`。四个重复 prompt 的测试只用于验证链路，不能用来估计真实 prompt 集的专家负载偏斜。

如需测 128 次 decode 前向，启动命令追加 `--min-output-len 128 --max-output-len 129 --max-batch-size-attn 128 --max-batch-size-expert 512`。导出时设 `--reference-steps 128` 可以逐 token 校验整段；若只导出了 4 步参考，则仅校验前 4 步，仍保存全部 128 个实测 token。正式启动后，worker 会按每张卡实际可用的 KV 块检查容量。真实权重模式还需要 embedding、LM head 和运行缓冲区，512 请求能否装下应以这次实测为准。

## 4. 固定到达率下的 decode 吞吐与延迟

`--paced-decode --rate R` 使第 `i` 个快照请求在基准开始后 `i / R` 秒到达，首个请求的编号为零。驱动按预定时刻投放，不等待上一个请求完成。每个快照请求只投放一次，512 条表示总请求数，同时运行多少条由到达率、执行速度及可用 KV 页共同决定。每卡的 attention 和 expert cap 限制单次调度的 batch 大小。省略 `--paced-decode` 仍运行前述整批统一起跑实验。

先在服务器的 `streamep4090` 环境中使用已验证的小快照，运行以下冒烟测试。命令假定当前目录是仓库根目录，且 `MODEL` 和 `RUN_DIR` 指向之前成功使用的模型和快照父目录。

```bash
export DMOE_RAY_TMPDIR=/workspace/zzc/ray
export RUN_TIMEOUT_SECONDS=1800
RESULT_DIR="${RUN_DIR:?请先设置 RUN_DIR}/paced-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RESULT_DIR"
common=(benchmark --paced-decode
  --checkpoint-path "${MODEL:?请先设置 MODEL}" --decode-snapshot "$RUN_DIR/snapshot"
  --gpu-usage 0.80 --max-batch-size-attn 128 --max-batch-size-expert 512)
bash experiments/scripts/rtx4090/launch.sh "${common[@]}" --rate 2 --file "$RESULT_DIR/check.csv"
```

正式比较时，在同一终端中将 `RUN_DIR` 指向 512 条请求的快照，并重新执行上面创建 `RESULT_DIR` 和 `common` 数组的几行。下面的示例对同一批输入依次测 4、8、12 req/s，每个请求执行 128 次 decode。每轮都单独启动引擎、预热和校验 greedy 参考前缀。

```bash
for rate in 4 8 12; do
  bash experiments/scripts/rtx4090/launch.sh "${common[@]}" \
    --rate "$rate" --arrival-window 10,40 \
    --min-output-len 128 --max-output-len 129 \
    --file "$RESULT_DIR/rate-${rate}.csv" || break
done
```

`--arrival-window 10,40` 统计首个计划到达后第 10 秒至第 40 秒之间的事件，包含左端点。结束时刻须小于等于 `N/R`，其中 `N` 为快照请求数。例如 512 条请求以 12 req/s 投放时可测至第 42.67 秒左右。省略该选项会统计整个投放窗口 `[0, N/R)`。根据各轮实际积压和吞吐判断所选窗口是否已经进入稳态，必要时增大样本数。四条请求的小快照只适合验证流程。

所有实验都从已完成 prefill 的状态开始。计时前每个 GPU worker 将不同 prompt 的 KV 暂存到该进程的主机内存，重复项复用同一份。以 512 条不同的 64-token prompt 为例，BF16 KV 约占每个 worker 3 GiB 主机内存。投放后按需取得 GPU 页、将 KV 从主机恢复到显存，再进入普通异步 decode 调度。排队和这次 KV 恢复均计入请求延迟。与其他系统比较时应对齐这个边界。

### 读懂结果

| CSV 列 | 统计口径 |
| --- | --- |
| `rate` | 计划到达率，单位为 req/s |
| `window_req_throughput` | 窗口内 driver 收到的完成请求数 / 窗口秒数 |
| `window_token_throughput` | 窗口内 detokenizer 收到的输出 token 数 / 窗口秒数，包含窗口结束时尚未完成的请求已产出的 token |
| `window_req_latency_median_ms`、`window_req_latency_p99_ms` | 对窗口内计划到达的请求，从计划到达到 driver 收到结果，追踪到全部完成，包含排队和结果轮询时间 |
| `window_outstanding_start`、`window_outstanding_end` | 窗口两端已到达、尚未向 driver 返回结果的请求数 |
| `window_waiting_admission_end` | 窗口结束时已经提交、仍在等待 DP 分配资源的请求数 |
| `req_throughput`、`token_throughput` | 从首次计划到达到最后一个响应的全程吞吐，包含停止投放后的排空时间 |

`*.paced.json` 保存本轮参数和每个请求的快照索引，并记录相对于起点的计划到达、实际提交、DP 分配、各输出 token 到达 detokenizer 和 driver 收到结果的时刻。用 `submitted_s - scheduled_s` 检查发压端是否跟得上预定速率，用 `admitted_s - submitted_s` 查看 DP 队列等待。driver 每轮查询后等待约 100 ms 再查询，分析短请求时可用 `response_s - token_times_s[-1]` 单独查看最后一个 token 到 driver 收到结果的延迟。另存的 `*.decode.json` 可核对输出 token 和参考前缀。观察接近饱和的速率时，应一起看窗口吞吐、未完成请求数变化和 P99 延迟。
