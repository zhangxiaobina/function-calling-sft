"""从两档轨迹 dump 构造 LlamaFactory DPO 偏好数据集（sharegpt + ranking）。

  chosen   来源 = base 档（轨迹成功、每跳真实调用工具）
  rejected 来源 = sft  档（同任务上早停 / 编造中间结果）
  只保留「base 成功 且 sft 失败」的任务 —— 这些对才真正承载
  『拿到部分数据后该继续调工具 vs 直接编造收尾』的偏好信号。

纯 Python3 标准库，无第三方依赖。用法：

  python build_dpo_dataset.py \
      --chosen   dpo_dumps/traj_base.jsonl \
      --rejected dpo_dumps/traj_sft.jsonl \
      --tools    dpo_dumps/tools.openai.json \
      --out      fc_dpo_train.jsonl \
      --mode     trajectory

两种 mode：
  trajectory（默认，最稳）：把整条轨迹折叠成单条 Hermes 风格文本作 chosen/rejected。
                          与计划 Step1 示例一致，LlamaFactory 原生支持（chosen/rejected 均为 gpt turn）。
  turn（更精准，实验性）：取 base/sft 工具调用序列的公共前缀作共享上文（function_call/observation
                          多轮塞进 conversations），只把第一处分歧之后的动作作 chosen/rejected。
                          偏好信号最干净；公共前缀为空时该任务自动回退 trajectory。
                          ⚠️ 用前务必在 LlamaFactory 里 `--do_train false` 预览一条，确认 template
                          对 conversations 中的 function_call/observation 多轮渲染正确。
"""
import argparse
import json


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _loads(x):
    """tool_calls 的 arguments 可能是 JSON 字符串，也可能已是 dict。"""
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _unwrap_tools(openai_tools):
    """edu-agent openai_tools() 形如 [{"type":"function","function":{...}}]，
    训练用 sharegpt tools 字段要的是 [{name,description,parameters}]（见 SFT 数据约定）。"""
    out = []
    for t in openai_tools:
        fn = t.get("function", t) if isinstance(t, dict) else t
        out.append(fn)
    return out


# --------------------------------------------------------------------------- #
# 轨迹 ↔ Hermes 文本
# --------------------------------------------------------------------------- #
def _system_of(messages):
    return next((m.get("content") for m in messages if m.get("role") == "system"), None)


def _fold(messages):
    """把 run_agent 的 OpenAI messages 折叠成单条 Hermes 风格 assistant 文本：
    每个工具调用 → <tool_call>{...}</tool_call>，每个工具返回 → <tool_response>...</tool_response>，
    助手文本原样拼接。跳过 system 与所有 user（query 单独进 conversations；nudge 在 max_nudges=0 时不存在）。"""
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                call = {"name": fn.get("name"), "arguments": _loads(fn.get("arguments"))}
                parts.append("<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>")
            if m.get("content"):
                parts.append(m["content"])
        elif role == "tool":
            parts.append("<tool_response>\n" + (m.get("content") or "") + "\n</tool_response>")
    return "\n".join(parts).strip()


def _call_names(messages):
    """按出现顺序取该轨迹的工具名序列。"""
    names = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                names.append(tc.get("function", {}).get("name"))
    return names


def _split_at(messages, k):
    """把 messages 在『第 k 个工具调用之后的工具返回』处切成 (前缀 turns, 剩余 messages)。
    前缀 turns 用 sharegpt 的 function_call/observation 表示，供 turn 模式塞进 conversations。
    剩余 messages 交给 _fold 作 chosen/rejected。"""
    prefix, calls_seen = [], 0
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if calls_seen >= k:
                break
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                prefix.append({"from": "function_call",
                               "value": json.dumps({"name": fn.get("name"),
                                                     "arguments": _loads(fn.get("arguments"))},
                                                    ensure_ascii=False)})
                calls_seen += 1
        elif role == "tool":
            prefix.append({"from": "observation", "value": m.get("content") or ""})
        elif role == "assistant" and m.get("content") and calls_seen >= k:
            break
        i += 1
    return prefix, messages[i:]


# --------------------------------------------------------------------------- #
# 构造单条偏好样本
# --------------------------------------------------------------------------- #
def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def make_sample(chosen_row, rejected_row, tools_str, mode):
    sys_val = _system_of(chosen_row["messages"])
    convs = []
    if sys_val:
        convs.append({"from": "system", "value": sys_val})
    convs.append({"from": "human", "value": chosen_row["query"]})

    if mode == "turn":
        k = _common_prefix_len(_call_names(chosen_row["messages"]),
                               _call_names(rejected_row["messages"]))
        if k > 0:  # 有公共前缀才走 turn；否则落到 trajectory
            prefix, c_rest = _split_at(chosen_row["messages"], k)
            _, r_rest = _split_at(rejected_row["messages"], k)
            convs.extend(prefix)
            chosen_val = _fold(c_rest)
            rejected_val = _fold(r_rest) or (rejected_row.get("final_answer") or "")
            return {"conversations": convs,
                    "chosen": {"from": "gpt", "value": chosen_val},
                    "rejected": {"from": "gpt", "value": rejected_val},
                    "tools": tools_str}

    # trajectory（默认 / turn 回退）
    chosen_val = _fold(chosen_row["messages"])
    rejected_val = _fold(rejected_row["messages"]) or (rejected_row.get("final_answer") or "")
    return {"conversations": convs,
            "chosen": {"from": "gpt", "value": chosen_val},
            "rejected": {"from": "gpt", "value": rejected_val},
            "tools": tools_str}


def main():
    ap = argparse.ArgumentParser(description="构造 LlamaFactory DPO 偏好数据集")
    ap.add_argument("--chosen", required=True, help="base 档 dump（chosen 来源）")
    ap.add_argument("--rejected", required=True, help="sft 档 dump（rejected 来源）")
    ap.add_argument("--tools", required=True, help="tools.openai.json")
    ap.add_argument("--out", default="fc_dpo_train.jsonl")
    ap.add_argument("--mode", choices=["trajectory", "turn"], default="trajectory")
    args = ap.parse_args()

    chosen = {r["id"]: r for r in _load_jsonl(args.chosen)}
    rejected = {r["id"]: r for r in _load_jsonl(args.rejected)}
    tools_str = json.dumps(_unwrap_tools(json.load(open(args.tools, encoding="utf-8"))),
                           ensure_ascii=False)

    samples, skipped = [], []
    for tid, c in chosen.items():
        r = rejected.get(tid)
        if r is None:
            skipped.append((tid, "rejected 档缺该任务"))
            continue
        if not c.get("success"):
            skipped.append((tid, "chosen(base)未成功，无正样本"))
            continue
        if r.get("success"):
            skipped.append((tid, "rejected(sft)也成功，无偏好差"))
            continue
        samples.append(make_sample(c, r, tools_str, args.mode))

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"mode={args.mode}  生成偏好对 {len(samples)} 条 → {args.out}")
    for tid, why in skipped:
        print(f"  跳过 {tid:32s} {why}")
    if len(samples) < 50:
        print(f"\n⚠️ 仅 {len(samples)} 条，远少于计划的 200–500。"
              "需扩充：对更多 multi_step 任务用强模型(通义/Qwen3-30B)生成 chosen、SFT 跑 rejected，"
              "人工核查 chosen 每跳真实调用、rejected 存在编造后并入本文件。")


if __name__ == "__main__":
    main()
