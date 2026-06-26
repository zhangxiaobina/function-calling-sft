#!/usr/bin/env python3
"""把 SFT 训练出的 LoRA adapter 合并回基座，得到一个独立的全权重模型目录。

为什么要 merge（而不是直接用 `--enable-lora` 在线挂载评测）：
    BFCL 的 `bfcl generate` 走的是 `QwenFCHandler`，它始终用 `model=<base 路径>`
    发请求，开 `--enable-lora/--lora-modules` 后请求并不会被路由到 LoRA 分支，
    导致 after == before（白跑）。最稳妥、不改 harness 的做法是先把 adapter
    `merge_and_unload` 进基座，base 与 merged 各用 `--local-model-path` 平等评测。

跑法（需要带 GPU / 足够内存，14B bf16 约需 ~30GB 显存或内存）:
    python scripts/merge_lora.py \
        --base    /path/to/Qwen3-14B \
        --adapter /path/to/qwen3-14b-fc-lora-adapter \
        --out     /path/to/Qwen3-14B-fc-merged

注意:
  - 14B bf16 merge 后约 28G，落盘前确保目标盘同时容得下 base+merged(≈56G)。
  - `low_cpu_mem_usage=False`: 完整读进内存而非 mmap 映射，避免删 base 后
    句柄仍占磁盘空间（详见 docs/training.md「踩坑」）。
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/path/to/Qwen3-14B", help="基座模型目录或 HF id")
    ap.add_argument("--adapter", default="/path/to/qwen3-14b-fc-lora-adapter", help="LoRA adapter 目录")
    ap.add_argument("--out", default="/path/to/Qwen3-14B-fc-merged", help="合并后输出目录")
    args = ap.parse_args()

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging...")
    model = model.merge_and_unload()

    print("Saving merged model...")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)

    print("Saving tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except Exception:
        # 某些 transformers 版本从 adapter 目录存 tokenizer 会崩，退回从基座取
        tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)

    print(f"Done. Output: {args.out}")


if __name__ == "__main__":
    main()
