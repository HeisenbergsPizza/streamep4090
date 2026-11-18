# 四张 RTX 4090 上运行 Qwen3-30B-A3B 真实 decode

实验用同一份本地 BF16 checkpoint。Transformers 计算一次 prompt prefill，导出每层 KV、首个输出 token 和一段 greedy 参考序列。StreamInfer 用自己的 Attention、CUTLASS expert 和异步调度执行后续 decode。当前入口支持单节点四卡、Attention DP4 / EP4 / TP1、等长 prompt、固定步数和 eager 执行。

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
