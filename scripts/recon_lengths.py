"""长度 recon：用 Qwen3 tokenizer 的 chat template(含 tools)估算每条样本的 token 长度，
据此选 cutoff_len。会下载 Qwen3-14B 的 tokenizer(仅分词器，非 28GB 权重)。

    python3 scripts/recon_lengths.py --file data/processed/fc_sft_train.jsonl --sample 20000
"""
from __future__ import annotations

import argparse
import json
import random


def sharegpt_to_messages(conv):
    """sharegpt(from/value) -> OpenAI 风格 messages(供 apply_chat_template)。
    把'助手解释文本 + 紧随的 function_call'合并成一条带 tool_calls 的 assistant 消息。"""
    messages = []
    for c in conv:
        frm, val = c["from"], c["value"]
        if frm == "system":
            messages.append({"role": "system", "content": val})
        elif frm == "human":
            messages.append({"role": "user", "content": val})
        elif frm == "gpt":
            messages.append({"role": "assistant", "content": val})
        elif frm == "observation":
            messages.append({"role": "tool", "content": val})
        elif frm == "function_call":
            calls = json.loads(val)
            if isinstance(calls, dict):
                calls = [calls]
            tool_calls = [
                {"type": "function", "function": {"name": c0["name"], "arguments": c0.get("arguments", {})}}
                for c0 in calls
            ]
            if messages and messages[-1]["role"] == "assistant" and "tool_calls" not in messages[-1]:
                messages[-1]["tool_calls"] = tool_calls
                messages[-1].setdefault("content", "")
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    return messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/processed/fc_sft_train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--sample", type=int, default=20000, help="抽样条数(0=全量)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.sample and len(rows) > args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)

    lengths = []
    n_fallback = 0
    for r in rows:
        try:
            msgs = sharegpt_to_messages(r["conversations"])
            tools = json.loads(r["tools"]) if r.get("tools") else None
            ids = tok.apply_chat_template(
                msgs, tools=tools, tokenize=True, add_generation_prompt=False, enable_thinking=False
            )
            lengths.append(len(ids))
        except Exception:
            n_fallback += 1
            text = (r.get("tools") or "") + "".join(c["value"] for c in r["conversations"])
            lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))

    lengths.sort()
    n = len(lengths)

    def pct(p):
        return lengths[min(n - 1, int(n * p))]

    print(f"# 样本={n} (fallback={n_fallback}), model={args.model}")
    print(f"min={lengths[0]} p50={pct(.5)} p90={pct(.9)} p95={pct(.95)} "
          f"p99={pct(.99)} p99.9={pct(.999)} max={lengths[-1]} mean={sum(lengths)//n}")
    print("\n# 候选 cutoff_len 覆盖率 / 截断率:")
    for c in (1024, 2048, 3072, 4096, 6144, 8192, 16384):
        over = sum(1 for x in lengths if x > c)
        print(f"  cutoff={c:>6}: 覆盖 {100*(n-over)/n:5.2f}%  截断 {over:>6} 条 ({100*over/n:.2f}%)")


if __name__ == "__main__":
    main()
