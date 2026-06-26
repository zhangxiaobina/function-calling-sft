"""下载后查数据源真实字段，用于复核/定稿各 adapter。流式拉取，只取前几条。

用法:
    python3 scripts/inspect_dataset.py --source glaive --limit 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import SOURCES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES.keys()))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--no-stream", action="store_true", help="关闭流式(数据集不支持流式时用)")
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = SOURCES[args.source]
    print(f"# {args.source} <- {cfg['hf_id']} (config={cfg['config']}, split={cfg['split']})")

    rows = []
    if not args.no_stream:
        try:
            ds = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)
            rows = list(itertools.islice(ds, args.limit))
        except Exception as e:  # noqa: BLE001
            print(f"# streaming 失败({type(e).__name__}: {e})，回退非流式", file=sys.stderr)
    if not rows:
        ds = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"])
        print(f"# n_rows = {len(ds)}")
        print(f"# features = {ds.features}")
        rows = [dict(ds[i]) for i in range(min(args.limit, len(ds)))]

    if rows:
        print(f"# columns = {list(rows[0].keys())}")
    print("\n# ---- samples ----")
    for i, row in enumerate(rows):
        print(f"\n## sample {i}")
        print(json.dumps(dict(row), ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
