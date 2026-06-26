"""构建统一 Function Calling SFT 训练集。

pipeline:  加载(HF) -> adapter 转 IR -> 校验 -> 类别推断 -> 去重 -> 切分 -> 序列化(LlamaFactory) -> 统计

- 核心 pipeline(adapt/validate/dedup/split/serialize)仅用标准库，可被 test_pipeline.py 离线复用。
- HF 下载只在 `--mode real` 时触发(需要 `pip install -r requirements.txt`)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters import ADAPTERS  # noqa: E402

# --------------------------------------------------------------------------- #
# 数据源登记。各源都是 HF 上的单/多文件(无加载脚本)，直接 hf_hub_download + 自解析，
# 绕开 datasets 4.x 对松散文件/config 的不稳。文件布局已用 list_repo_files 核实。
# --------------------------------------------------------------------------- #
SOURCES: Dict[str, Dict[str, Any]] = {
    "hermes_reasoning": {  # 主力 51k, parquet, 独立 tools 列(xLAM 扁平参数) + scenario_category
        "hf_id": "interstellarninja/hermes_reasoning_tool_use",
        "files": ["data/train-00000-of-00001.parquet"], "reader": "parquet",
        "adapter": "hermes", "adapter_kwargs": {"strip_think": True},  # v1: 统一非思考
    },
    "hermes_v1": {  # Hermes 标准格式, 独立 tools 列 + category
        "hf_id": "NousResearch/hermes-function-calling-v1",
        "files": ["func-calling.json", "func-calling-singleturn.json"], "reader": "json_array",
        "adapter": "hermes",
    },
    "toolace": {  # Python 伪调用串 + type:dict
        "hf_id": "Team-ACE/ToolACE",
        "files": ["data.json"], "reader": "json_array",
        "adapter": "toolace",
    },
    "glaive": {  # 铺量, system+chat 拼接串; 限量 40k(最简单的单轮数据, 不让它盖过其他)
        "hf_id": "glaiveai/glaive-function-calling-v2",
        "files": ["glaive-function-calling-v2.json"], "reader": "json_array",
        "adapter": "glaive", "limit": 40000,
    },
}

_VALID_ROLES = {"user", "assistant", "tool_calls", "tool"}


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate_record(ir: Dict[str, Any], max_chars: int = 0) -> Tuple[bool, str]:
    if not ir:
        return False, "empty"
    turns = ir.get("turns") or []
    if not turns:
        return False, "no_turns"
    roles = [t.get("role") for t in turns]
    if any(r not in _VALID_ROLES for r in roles):
        return False, "bad_role"
    if "user" not in roles:
        return False, "no_user"
    if not any(r in ("assistant", "tool_calls") for r in roles):
        return False, "no_assistant"

    tools = ir.get("tools") or []
    tool_names = set()
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            return False, "bad_tool_schema"
        params = t.get("parameters")
        if not isinstance(params, dict) or "properties" not in params:
            return False, "bad_tool_params"
        tool_names.add(t["name"])

    seen_call = False
    for i, t in enumerate(turns):
        if t["role"] == "tool_calls":
            calls = t.get("calls") or []
            if not calls:
                return False, "empty_calls"
            for c in calls:
                if not c.get("name"):
                    return False, "call_no_name"
                if not isinstance(c.get("arguments"), dict):
                    return False, "args_not_dict"
                # 调用必须引用已定义的工具(否则在教模型瞎调)
                if tool_names and c["name"] not in tool_names:
                    return False, "call_undefined_tool"
            seen_call = True
        if t["role"] == "tool" and not seen_call:
            return False, "tool_before_call"

    if max_chars:
        if len(json.dumps(ir, ensure_ascii=False)) > max_chars:
            return False, "too_long"
    return True, "ok"


# --------------------------------------------------------------------------- #
# 类别推断（用于分层切分 + 统计；对齐 BFCL 思路）
# --------------------------------------------------------------------------- #
def infer_category(ir: Dict[str, Any]) -> str:
    if ir.get("category") and ir["category"] != "unknown":
        return ir["category"]
    turns = ir["turns"]
    call_turns = [t for t in turns if t["role"] == "tool_calls"]
    n_user = sum(1 for t in turns if t["role"] == "user")
    if not call_turns:
        return "no_call"  # relevance / irrelevant
    if any(len(t.get("calls", [])) > 1 for t in call_turns):
        return "parallel"
    if n_user > 1:
        return "multiturn"
    if len(call_turns) > 1:
        return "multistep"
    return "single"


# --------------------------------------------------------------------------- #
# 去重
# --------------------------------------------------------------------------- #
def record_hash(ir: Dict[str, Any]) -> str:
    key = {
        "system": ir.get("system"),
        "tools": sorted([t["name"] for t in ir.get("tools", [])]),
        "turns": [
            (t["role"], t.get("content"), t.get("calls"))
            for t in ir.get("turns", [])
        ],
    }
    blob = json.dumps(key, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 序列化 -> LlamaFactory sharegpt + tools
# --------------------------------------------------------------------------- #
def to_llamafactory(ir: Dict[str, Any], parallel_as_array: bool = True) -> Dict[str, Any]:
    conv: List[Dict[str, str]] = []
    if ir.get("system"):
        conv.append({"from": "system", "value": ir["system"]})
    for t in ir["turns"]:
        role = t["role"]
        if role == "user":
            conv.append({"from": "human", "value": t["content"]})
        elif role == "assistant":
            conv.append({"from": "gpt", "value": t["content"]})
        elif role == "tool":
            conv.append({"from": "observation", "value": t["content"]})
        elif role == "tool_calls":
            calls = t["calls"]
            if len(calls) == 1 and not parallel_as_array:
                value = json.dumps(calls[0], ensure_ascii=False)
            elif len(calls) == 1:
                value = json.dumps(calls[0], ensure_ascii=False)
            else:
                value = json.dumps(calls, ensure_ascii=False)  # 并行调用
            conv.append({"from": "function_call", "value": value})
    tools_str = json.dumps(ir.get("tools", []), ensure_ascii=False)
    return {"conversations": conv, "tools": tools_str}


_LF_TAGS = {
    "role_tag": "from", "content_tag": "value",
    "user_tag": "human", "assistant_tag": "gpt",
    "observation_tag": "observation", "function_tag": "function_call",
    "system_tag": "system",
}


def dataset_info(train_file: str, dev_file: str) -> Dict[str, Any]:
    """生成 LlamaFactory 的 dataset_info.json(放在 out_dir，dataset_dir 指向它)。"""
    common = {
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "tools": "tools"},
        "tags": _LF_TAGS,
    }
    return {
        "fc_sft": {"file_name": os.path.basename(train_file), **common},
        "fc_sft_dev": {"file_name": os.path.basename(dev_file), **common},
    }


# --------------------------------------------------------------------------- #
# pipeline 主流程（输入: (source, raw_row) 迭代器；输出: 统计 + 写文件）
# --------------------------------------------------------------------------- #
def run_pipeline(
    raw_iter: Iterable[Tuple[str, Dict[str, Any]]],
    out_dir: str,
    dev_ratio: float = 0.01,
    dev_cap: int = 500,
    seed: int = 42,
    max_chars: int = 0,
    write: bool = True,
) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    seen_hashes = set()
    drop_reasons: Counter = Counter()
    cat_counter: Counter = Counter()
    src_counter: Counter = Counter()
    n_in = 0
    n_dup = 0

    for source, raw in raw_iter:
        n_in += 1
        cfg = SOURCES.get(source, {})
        adapter = ADAPTERS[cfg.get("adapter", "hermes")]
        kwargs = cfg.get("adapter_kwargs", {})
        try:
            ir = adapter(raw, source=source, **kwargs)
        except Exception as e:  # adapter 不该崩，崩了记下来
            drop_reasons[f"adapter_error:{type(e).__name__}"] += 1
            continue
        if ir is None:
            drop_reasons["adapter_none"] += 1
            continue
        ok, reason = validate_record(ir, max_chars=max_chars)
        if not ok:
            drop_reasons[reason] += 1
            continue
        h = record_hash(ir)
        if h in seen_hashes:
            n_dup += 1
            continue
        seen_hashes.add(h)
        ir["category"] = infer_category(ir)
        kept.append(ir)
        cat_counter[ir["category"]] += 1
        src_counter[source] += 1

    # 分层切分(按 source+category)，固定 seed
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for ir in kept:
        buckets.setdefault(f'{ir["source"]}|{ir["category"]}', []).append(ir)
    dev: List[Dict[str, Any]] = []
    train: List[Dict[str, Any]] = []
    for _, items in sorted(buckets.items()):
        rng.shuffle(items)
        k = min(int(len(items) * dev_ratio), max(1, dev_cap // max(1, len(buckets))))
        dev.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(dev)

    stats = {
        "n_in": n_in,
        "n_kept": len(kept),
        "n_dup": n_dup,
        "n_train": len(train),
        "n_dev": len(dev),
        "by_source": dict(src_counter),
        "by_category": dict(cat_counter),
        "drop_reasons": dict(drop_reasons),
    }

    if write:
        os.makedirs(out_dir, exist_ok=True)
        train_path = os.path.join(out_dir, "fc_sft_train.jsonl")
        dev_path = os.path.join(out_dir, "fc_sft_dev.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for ir in train:
                f.write(json.dumps(to_llamafactory(ir), ensure_ascii=False) + "\n")
        with open(dev_path, "w", encoding="utf-8") as f:
            for ir in dev:
                f.write(json.dumps(to_llamafactory(ir), ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info(train_path, dev_path), f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        stats["_out_dir"] = out_dir
    return stats


# --------------------------------------------------------------------------- #
# 真实数据加载：hf_hub_download 指定文件 + 自解析；只在 --mode real 调用。
# --------------------------------------------------------------------------- #
def _read_file(path: str, reader: str):
    if reader == "json_array":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):  # 兜底：可能包成 {"data": [...]}
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        for row in data:
            yield row
    elif reader == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                yield row
    else:
        raise ValueError(f"unknown reader: {reader}")


def iter_files(sources: List[str], limit_per_source: int = 0):
    from huggingface_hub import hf_hub_download

    for source in sources:
        cfg = SOURCES[source]
        cap = limit_per_source or cfg.get("limit", 0)  # 全局 --limit 优先，否则用源自带 limit
        count = 0
        for fn in cfg["files"]:
            print(f"[load] {source} <- {cfg['hf_id']}::{fn}"
                  + (f" (cap {cap})" if cap else ""), file=sys.stderr)
            path = hf_hub_download(cfg["hf_id"], fn, repo_type="dataset")
            for row in _read_file(path, cfg["reader"]):
                if cap and count >= cap:
                    break
                yield source, row
                count += 1
            if cap and count >= cap:
                break


def iter_fixtures(sources: List[str], fixtures_dir: str):
    for source in sources:
        path = os.path.join(fixtures_dir, f"{source}_sample.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield source, json.loads(line)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="构建 Function Calling SFT 训练集")
    ap.add_argument("--mode", choices=["real", "fixtures"], default="real")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES.keys()))
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--fixtures-dir", default="data/fixtures")
    ap.add_argument("--limit-per-source", type=int, default=0, help="每源最多取几条(0=全部)")
    ap.add_argument("--dev-ratio", type=float, default=0.01)
    ap.add_argument("--dev-cap", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=0, help="超长样本丢弃阈值(0=不丢，仅训练前 token recon 用)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "real":
        raw_iter = iter_files(args.sources, args.limit_per_source)
    else:
        raw_iter = iter_fixtures(args.sources, args.fixtures_dir)

    stats = run_pipeline(
        raw_iter, out_dir=args.out, dev_ratio=args.dev_ratio,
        dev_cap=args.dev_cap, seed=args.seed, max_chars=args.max_chars,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
