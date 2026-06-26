"""按类别分层下采样 fc_sft_train.jsonl → 更小的训练集(缩短 SFT 时长)。

- 类别从对话结构重推(复用 build_dataset.infer_category 的等价逻辑),
  因为序列化后的 jsonl 已无 source/category 字段。
- 小而难的类别(parallel/multistep)高保留,铺量大类(single/no_call/multiturn)降采样,
  保证 BFCL 五类覆盖不丢。固定 seed=42 可复现。

用法:
  python scripts/subsample.py --in data/processed/fc_sft_train.jsonl --stats   # 只看分布
  python scripts/subsample.py --in data/processed/fc_sft_train.jsonl \
         --out data/processed/fc_sft_train_35k.jsonl                            # 写子集
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict


def conv_to_category(rec: dict) -> str:
    """从 LlamaFactory sharegpt 记录重推 BFCL 类别(等价 build_dataset.infer_category)。"""
    conv = rec.get("conversations", [])
    call_turns = []
    n_user = 0
    for t in conv:
        frm = t.get("from")
        if frm == "human":
            n_user += 1
        elif frm == "function_call":
            # value 可能是单个 dict 或并行调用的 array
            try:
                parsed = json.loads(t.get("value", "null"))
            except json.JSONDecodeError:
                parsed = None
            n_calls = len(parsed) if isinstance(parsed, list) else 1
            call_turns.append(n_calls)
    if not call_turns:
        return "no_call"
    if any(n > 1 for n in call_turns):
        return "parallel"
    if n_user > 1:
        return "multiturn"
    if len(call_turns) > 1:
        return "multistep"
    return "single"


# 每类目标保留数(None = 全留)。小难类全留,大铺量类降采样。
# 类别由对话结构重推:relevance 无调用→并入 no_call;parallel=任一轮多调用。
DEFAULT_CAPS = {
    "multistep": None,    # ~4.4k 全留(最稀少且难,FC 亮点)
    "parallel": 6000,     # 从 ~10k 收到 6k(难类,保强覆盖但不过重)
    "no_call": 7000,      # 含 relevance,保住"不乱调用"的负样本信号
    "single": 7000,
    "multiturn": 10000,   # 多轮 FC 主力,留最多
}
# 预期总量 ≈ 4.4k+6k+7k+7k+10k ≈ 34.4k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats", action="store_true", help="只打印类别分布,不写文件")
    args = ap.parse_args()

    by_cat: dict[str, list[dict]] = defaultdict(list)
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_cat[conv_to_category(rec)].append(rec)

    full = Counter({c: len(v) for c, v in by_cat.items()})
    print("=== 全量类别分布 ===")
    for c, n in full.most_common():
        print(f"  {c:12s} {n}")
    print(f"  {'TOTAL':12s} {sum(full.values())}")

    if args.stats:
        return

    rng = random.Random(args.seed)
    out_recs: list[dict] = []
    kept = Counter()
    for cat, items in by_cat.items():
        cap = DEFAULT_CAPS.get(cat)
        if cap is None or cap >= len(items):
            sel = items
        else:
            sel = rng.sample(items, cap)
        kept[cat] = len(sel)
        out_recs.extend(sel)
    rng.shuffle(out_recs)

    print("\n=== 子集类别分布 ===")
    for c, n in kept.most_common():
        print(f"  {c:12s} {n}")
    print(f"  {'TOTAL':12s} {sum(kept.values())}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n写出 {len(out_recs)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
