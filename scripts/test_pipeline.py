"""离线自测：用 data/fixtures 的合成小样本(模拟各源真实格式)跑通整条 pipeline 逻辑。

    python3 scripts/test_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
FIX = os.path.join(ROOT, "data", "fixtures")

from adapters import adapt_glaive, adapt_hermes_conversations, adapt_toolace  # noqa: E402
from build_dataset import (  # noqa: E402
    infer_category,
    iter_fixtures,
    run_pipeline,
    to_llamafactory,
    validate_record,
)

_FAILS = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILS.append(msg)


def load(name):
    rows = []
    with open(os.path.join(FIX, f"{name}_sample.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cat(ir):
    return infer_category(ir)


def main() -> None:
    print("== glaive (system+chat 拼接串, 单引号嵌套 arguments) ==")
    g = load("glaive")
    g0 = adapt_glaive(g[0])
    calls = [t for t in g0["turns"] if t["role"] == "tool_calls"]
    check(len(calls) == 1 and calls[0]["calls"][0]["name"] == "get_stock_price", "row0: 工具调用解析")
    check(calls[0]["calls"][0]["arguments"] == {"symbol": "AAPL"},
          f"row0: 单引号嵌套 arguments 正确 (got {calls[0]['calls'][0]['arguments']})")
    check(g0["tools"][0]["parameters"].get("type") == "object", "row0: 工具 parameters 为标准 object")
    check(cat(adapt_glaive(g[1])) == "no_call", "row1: 写诗→no_call")
    check(cat(adapt_glaive(g[2])) == "multiturn", "row2: 两轮→multiturn")

    print("== hermes_v1 (独立 tools 列, 标准 <tool_call>) ==")
    h = load("hermes_v1")
    h0 = adapt_hermes_conversations(h[0], source="hermes_v1")
    check(len(h0["tools"]) == 1, "row0: 提取 1 个工具(避开说明文字里的空 <tools>)")
    check(cat(h0) == "single", f"row0: single (got {cat(h0)})")
    h1 = adapt_hermes_conversations(h[1], source="hermes_v1")
    pcalls = [t for t in h1["turns"] if t["role"] == "tool_calls"]
    check(len(pcalls) == 1 and len(pcalls[0]["calls"]) == 2, "row1: 一轮两并行调用")
    check(cat(h1) == "parallel", f"row1: parallel (got {cat(h1)})")

    print("== hermes_reasoning (xLAM 扁平参数 + scenario_category + <think>) ==")
    r = load("hermes_reasoning")
    r0 = adapt_hermes_conversations(r[0], source="hermes_reasoning")
    p = r0["tools"][0]["parameters"]
    check(p.get("type") == "object" and "properties" in p and "text" in p["properties"],
          f"扁平参数已包成标准 JSON Schema (got {p})")
    check(p["properties"]["text"].get("type") == "string", "Python 类型 str→string 归一")
    check(p.get("required") == ["text"], f"无 default 的参数推断为 required (got {p.get('required')})")
    check(r0["category"] == "single", f"采用 scenario_category=single (got {r0['category']})")
    asst = [t for t in r0["turns"] if t["role"] == "assistant"]
    check(any("<think>" in t["content"] for t in asst), "默认保留 <think> 推理")
    r0s = adapt_hermes_conversations(r[0], source="hermes_reasoning", strip_think=True)
    check(all("<think>" not in t["content"] for t in r0s["turns"] if t["role"] == "assistant"),
          "strip_think=True 去掉推理")

    print("== toolace (Python 伪调用串 + type:dict) ==")
    t = load("toolace")
    t0 = adapt_toolace(t[0])
    tc = [x for x in t0["turns"] if x["role"] == "tool_calls"]
    check(len(tc) == 1 and tc[0]["calls"][0]["name"] == "get_market_trends", "row0: 伪调用 name 解析")
    check(tc[0]["calls"][0]["arguments"] == {"trend_type": "MARKET_INDEXES", "country": "us"},
          f"row0: 伪调用 arguments 解析 (got {tc[0]['calls'][0]['arguments']})")
    tp = t0["tools"][0]["parameters"]
    check(tp.get("type") == "object", "row0: parameters type:dict→object")
    check("country" not in tp.get("required", []) and "trend_type" in tp.get("required", []),
          f"row0: 保留 required, 带 default 的不算 required (got {tp.get('required')})")
    check(t0["system"] is None, "row0: 工具单列后 system 置空")
    check(cat(adapt_toolace(t[1])) == "no_call", "row1: 拒答→no_call")
    t2 = adapt_toolace(t[2])  # 函数名自带括号
    t2c = [x for x in t2["turns"] if x["role"] == "tool_calls"]
    check(t2c and t2c[0]["calls"][0]["name"] == "User Feed (Video Posts) V2",
          f"row2: 名字含括号也能完整解析 (got {t2c[0]['calls'][0]['name'] if t2c else None})")
    check(t2c and t2c[0]["calls"][0]["arguments"] == {"user_id": "12345"}, "row2: 含括号名的 arguments 正确")

    print("== 校验规则 ==")
    bad = {"source": "x", "system": None,
           "tools": [{"name": "a", "description": "", "parameters": {"type": "object", "properties": {}}}],
           "turns": [{"role": "user", "content": "hi"},
                     {"role": "tool_calls", "calls": [{"name": "ghost", "arguments": {}}]}]}
    ok, reason = validate_record(bad)
    check((not ok) and reason == "call_undefined_tool", f"调用未定义工具应丢弃 (got {ok},{reason})")

    print("== 序列化 LlamaFactory ==")
    lf = to_llamafactory(h1)  # parallel
    fc = [c for c in lf["conversations"] if c["from"] == "function_call"]
    check(len(fc) == 1 and isinstance(json.loads(fc[0]["value"]), list), "并行调用→JSON 数组")
    check(isinstance(json.loads(lf["tools"]), list), "tools 是合法 JSON 数组字符串")
    check({c["from"] for c in lf["conversations"]} <= {"system", "human", "gpt", "function_call", "observation"},
          "sharegpt 角色合法")

    print("== 整体 pipeline 统计 (fixtures) ==")
    stats = run_pipeline(iter_fixtures(["glaive", "hermes_v1", "hermes_reasoning", "toolace"], FIX),
                         out_dir="", write=False)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    check(stats["n_kept"] == 9, f"保留 9 条 (got {stats['n_kept']})")
    check(not stats["drop_reasons"], f"无丢弃 (got {stats['drop_reasons']})")
    check(stats["by_category"].get("parallel") == 1, "1 条 parallel")
    check(stats["by_category"].get("no_call") == 2, "2 条 no_call(写诗 + 拒答)")
    check(stats["by_category"].get("multiturn") == 1, "1 条 multiturn")

    print()
    if _FAILS:
        print(f"❌ {len(_FAILS)} 个断言失败")
        sys.exit(1)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
