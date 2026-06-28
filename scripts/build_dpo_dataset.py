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
# 配对判据：strict（精确成功）/ loose（覆盖度——B）
# --------------------------------------------------------------------------- #
def _match_group(group, name):
    """required 每项是工具名 str 或 list[str]（任一可接受）。"""
    return name in (group if isinstance(group, list) else [group])


def _coverage(trace, required):
    """该轨迹在『必需工具有序子序列』上推进到第几步（0..len(required)）——衡量它把真实
    工具链走了多远。required 缺失时返回 None（回退 strict）。"""
    if not required:
        return None
    tools = [t.get("tool") for t in (trace or [])]
    i = 0
    for t in tools:
        if i < len(required) and _match_group(required[i], t):
            i += 1
    return i


def is_pair(c, r, criterion):
    """c=base 记录(chosen 源), r=sft 记录(rejected 源)。返回是否构成偏好对。

    strict：base 严格成功 且 sft 未成功（原口径）。
    loose（B）：在 strict 基础上，再纳入『base 把真实工具链走得比 sft 更远』的对——
      base 干净(无报错、有最终回答、覆盖≥2)且 sft 编造/早停(未成功、有最终回答)且
      base 覆盖度 > sft 覆盖度。捕获 base 用了同样合理的工具但没精确命中钦定序列的情形，
      同时靠『覆盖度更高』过滤掉 base 自己也跑偏/死循环的伪对。
    """
    if r is None:
        return False, "rejected 档缺该任务"
    strict = bool(c.get("success")) and not bool(r.get("success"))
    if criterion == "strict":
        return strict, ("" if strict else "未满足 strict(base成功∧sft失败)")
    # loose
    if strict:
        return True, ""
    required = c.get("required_tools")
    bc, sc = _coverage(c.get("trace"), required), _coverage(r.get("trace"), required)
    if bc is None:
        return False, "无 required_tools，loose 无法判(请重 dump)"
    base_good = (not c.get("error")) and bool(c.get("final_answer")) and bc >= 2
    sft_bad = (not r.get("success")) and bool(r.get("final_answer"))
    if base_good and sft_bad and bc > sc:
        return True, ""
    return False, f"loose 未过(base_cov={bc} sft_cov={sc} base_good={base_good} sft_bad={sft_bad})"


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
    ap.add_argument("--criterion", choices=["strict", "loose"], default="strict",
                    help="strict=base精确成功∧sft失败(原口径)；"
                         "loose(B)=再纳入 base 真实工具链覆盖度高于 sft 的对(需 dump 带 required_tools)")
    args = ap.parse_args()

    chosen = {r["id"]: r for r in _load_jsonl(args.chosen)}
    rejected = {r["id"]: r for r in _load_jsonl(args.rejected)}
    tools_str = json.dumps(_unwrap_tools(json.load(open(args.tools, encoding="utf-8"))),
                           ensure_ascii=False)

    samples, skipped = [], []
    for tid, c in chosen.items():
        r = rejected.get(tid)
        ok, why = is_pair(c, r, args.criterion)
        if not ok:
            skipped.append((tid, why))
            continue
        samples.append(make_sample(c, r, tools_str, args.mode))

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"mode={args.mode} criterion={args.criterion}  生成偏好对 {len(samples)} 条 → {args.out}")
    for tid, why in skipped:
        print(f"  跳过 {tid:32s} {why}")
    if len(samples) < 50:
        print(f"\n⚠️ 仅 {len(samples)} 条，远少于计划的 200–500。"
              "需扩充：对更多 multi_step 任务用强模型(通义/Qwen3-30B)生成 chosen、SFT 跑 rejected，"
              "人工核查 chosen 每跳真实调用、rejected 存在编造后并入本文件。")


if __name__ == "__main__":
    main()
