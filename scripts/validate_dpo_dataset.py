"""本地校验 fc_dpo_train.jsonl 是否符合 LlamaFactory sharegpt + ranking(DPO) 规范，
在上 GPU 跑 LlamaFactory 前先挡掉格式坑，省一轮 PGX 往返。纯标准库，无需装 LlamaFactory。

  python validate_dataset.py fc_dpo_train.jsonl [--cutoff 8192]

校验项：字段齐全 / 上文角色合法（不含 gpt）/ system 仅在开头 / 末条为 human|observation /
       chosen,rejected 均为非空 gpt turn / chosen≠rejected / tools 为合法 JSON 数组 / 长度粗估。
退出码非 0 表示存在 error。
"""
import argparse
import json
import sys

ALLOWED_CTX_ROLES = {"system", "human", "function_call", "observation"}


def _estimate_tokens(text: str) -> int:
    # 粗估上界：中英混合 + JSON，按 ~0.6 token/字符（中文偏高、JSON 英文偏低，取保守系数）。
    # 仅用于「会不会接近 cutoff」的早期预警，精确长度以 LlamaFactory 预处理为准。
    return int(len(text or "") * 0.6)


def validate_row(row: dict, cutoff: int):
    errs, warns = [], []
    for k in ("conversations", "chosen", "rejected", "tools"):
        if k not in row:
            errs.append(f"缺字段 {k}")
    if errs:
        return errs, warns

    convs = row["conversations"]
    if not isinstance(convs, list) or not convs:
        errs.append("conversations 应为非空 list")
    else:
        for j, m in enumerate(convs):
            frm = m.get("from")
            if frm not in ALLOWED_CTX_ROLES:
                errs.append(f"conversations[{j}].from='{frm}' 非法（上文不应含 gpt，gpt 放 chosen/rejected）")
            if frm == "system" and j != 0:
                errs.append(f"system 应在开头，却出现在 conversations[{j}]")
        if convs[-1].get("from") not in {"human", "observation"}:
            errs.append(f"conversations 末条 from='{convs[-1].get('from')}'，应为 human 或 observation")

    for k in ("chosen", "rejected"):
        m = row[k]
        if not isinstance(m, dict) or m.get("from") != "gpt":
            errs.append(f"{k}.from 应为 gpt")
        elif not (m.get("value") or "").strip():
            errs.append(f"{k}.value 为空")
    if row.get("chosen") == row.get("rejected"):
        errs.append("chosen == rejected，无偏好信号")

    if not isinstance(row["tools"], str):
        errs.append("tools 应为 JSON 字符串")
    else:
        try:
            if not isinstance(json.loads(row["tools"]), list):
                errs.append("tools parse 后应为 list")
        except json.JSONDecodeError:
            errs.append("tools 不是合法 JSON 字符串")

    # 长度粗估：prompt(上文+tools) + 较长的一支回复
    ctx = "".join(m.get("value", "") for m in convs) if isinstance(convs, list) else ""
    longest = max(_estimate_tokens(row["chosen"].get("value", "")) if isinstance(row["chosen"], dict) else 0,
                  _estimate_tokens(row["rejected"].get("value", "")) if isinstance(row["rejected"], dict) else 0)
    approx = _estimate_tokens(ctx) + _estimate_tokens(row["tools"]) + longest
    if approx > cutoff * 0.9:
        warns.append(f"粗估 ~{approx} token，接近/超过 cutoff {cutoff}，可能被截断（精确以 LlamaFactory 为准）")
    return errs, warns


def main():
    ap = argparse.ArgumentParser(description="校验 LlamaFactory DPO 数据集")
    ap.add_argument("path", help="fc_dpo_train.jsonl")
    ap.add_argument("--cutoff", type=int, default=8192, help="与训练 yaml 的 cutoff_len 一致")
    args = ap.parse_args()

    n, n_bad = 0, 0
    total_warns = 0
    with open(args.path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [行{ln}] ✗ 非法 JSON: {e}")
                n_bad += 1
                continue
            errs, warns = validate_row(row, args.cutoff)
            tid = row.get("conversations", [{}])
            tag = f"行{ln}"
            if errs:
                n_bad += 1
                for e in errs:
                    print(f"  [{tag}] ✗ {e}")
            for w in warns:
                total_warns += 1
                print(f"  [{tag}] ⚠ {w}")

    print(f"\n共 {n} 条；{n_bad} 条有 error，{total_warns} 条警告。")
    if n == 0:
        print("⚠️ 空文件——还没 build 出真实数据？")
        sys.exit(1)
    if n_bad:
        print("✗ 存在 error，修正后再上 LlamaFactory。")
        sys.exit(1)
    print("✓ 格式校验通过，可上 LlamaFactory。建议仍先 `--do_train false` 预览一条确认 template 渲染。")


if __name__ == "__main__":
    main()
