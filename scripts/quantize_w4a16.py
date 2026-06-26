#!/usr/bin/env python3
"""GPTQ W4A16 量化：把 merged 的 14B bf16 模型压成 4-bit 权重 / 16-bit 激活。

方法学:
  - 用 llm-compressor 的 `GPTQModifier(scheme="W4A16")`，只量化线性层、保留 `lm_head`。
  - 校准集用 **in-domain 的 FC 训练数据**，比通用语料更贴工具调用时的激活分布，量化掉点更小。
  - W4A16（权重 4bit、激活仍 16bit）对解码是带宽友好的：权重体积大幅缩小，单流解码近线性提速；
    激活不量化，AST 精度损失极小。

跑法（带 GPU 的环境）:
    python scripts/quantize_w4a16.py \
        --model /path/to/Qwen3-14B-fc-merged \
        --out   /path/to/Qwen3-14B-fc-merged-W4A16 \
        --calib data/processed/fc_sft_train.jsonl

提示:
  - `llm-compressor` 与 `transformers` 版本需匹配，否则 import 失败。
  - 统一内存架构（如 GB10 / aarch64）上，量化工具逐层打印显存占用时可能调用不被支持的 nvml 接口而报错，
    需给该日志打补丁（try/except 兜住）才能跑通；这只是日志、不影响量化本身。详见 docs/quantization.md。
"""
import argparse
import json

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

# sharegpt role -> chat role；observation(工具返回) 折成 user 的 "[tool result] ..."
RMAP = {"system": "system", "human": "user", "gpt": "assistant", "function_call": "assistant"}


def build_calib_texts(tok, calib_path: str, num: int, maxlen: int) -> list[str]:
    texts: list[str] = []
    with open(calib_path) as f:
        for line in f:
            if len(texts) >= num:
                break
            r = json.loads(line)
            try:
                tools = json.loads(r["tools"]) if isinstance(r.get("tools"), str) else r.get("tools")
            except Exception:
                tools = None
            msgs = []
            for m in r["conversations"]:
                role = RMAP.get(m["from"])
                if role is None:  # observation -> 折成 user 的工具返回
                    msgs.append({"role": "user", "content": "[tool result] " + str(m["value"])})
                else:
                    msgs.append({"role": role, "content": str(m["value"])})
            try:
                t = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
            except Exception:
                try:
                    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    continue
            texts.append(t)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/path/to/Qwen3-14B-fc-merged")
    ap.add_argument("--out", default="/path/to/Qwen3-14B-fc-merged-W4A16")
    ap.add_argument("--calib", default="data/processed/fc_sft_train.jsonl", help="in-domain 校准集(jsonl)")
    ap.add_argument("--num", type=int, default=512, help="校准样本数")
    ap.add_argument("--maxlen", type=int, default=2048, help="校准最大序列长度")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = build_calib_texts(tok, args.calib, args.num, args.maxlen)
    print("校准样本数:", len(texts))

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(
        lambda e: tok(e["text"], truncation=True, max_length=args.maxlen, add_special_tokens=False),
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.maxlen,
        num_calibration_samples=len(texts),
        output_dir=args.out,
    )
    tok.save_pretrained(args.out)
    print("DONE W4A16 ->", args.out)


if __name__ == "__main__":
    main()
