#!/usr/bin/env python3
"""解码吞吐基准：同机对比 bf16 与 W4A16 在不同 batch 下的 tok/s。

固定生成 512 token（`ignore_eos`）以剔除提前停止带来的噪声，先 warmup 再计时。
W4A16 在低 batch（带宽受限）提速最大、接近权重压缩比；高 batch 转 compute 受限
（Marlin 反量化回 fp16 的算力成本显现）后提速收窄——这条曲线本身就是量化的体检。

跑法（在带 GPU 的 vLLM 环境，逐个模型各跑一次）:
    python scripts/bench_decode.py /path/to/Qwen3-14B-fc-merged       merged_bf16
    python scripts/bench_decode.py /path/to/Qwen3-14B-fc-merged-W4A16 w4a16
"""
import sys
import time

from vllm import LLM, SamplingParams

model_path = sys.argv[1]
label = sys.argv[2]

MAX_LEN = 2048
OUT_TOKENS = 512
PROMPT = "Explain in detail, step by step, how a transformer neural network processes a sequence of tokens."

llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.85,
    max_model_len=MAX_LEN,
    enforce_eager=False,
    disable_log_stats=True,
)


def bench(batch):
    sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOKENS, ignore_eos=True)
    prompts = [PROMPT] * batch
    # warmup (compile/cuda-graph already done at load; this stabilizes timing)
    llm.generate(prompts, sp, use_tqdm=False)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    return out_tok, dt, out_tok / dt


print("\n========== %s :: %s ==========" % (label, model_path), flush=True)
for b in [1, 8, 64]:
    out_tok, dt, tps = bench(b)
    print(
        "[RESULT] %s batch=%-3d out_tokens=%-6d time=%6.2fs  agg_throughput=%8.1f tok/s  per_req=%7.1f tok/s"
        % (label, b, out_tok, dt, tps, tps / b),
        flush=True,
    )
