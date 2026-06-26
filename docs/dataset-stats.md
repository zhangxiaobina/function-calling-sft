# 数据集统计（v1）

> 由 `scripts/build_dataset.py` 构建。原始数据不入库，复现见下。

## 复现

```bash
pip install -r requirements.txt
python3 scripts/build_dataset.py --mode real --out data/processed
# 产出: data/processed/fc_sft_train.jsonl + fc_sft_dev.jsonl + dataset_info.snippet.json + build_stats.json
```

## v1 结果（2026-06-23）

- 输入 106,090 → **保留 93,078**；去重 10,164；验证丢弃 2,848（2.7%）
- train 92,734 / dev 344（按 source+category 分层，seed=42）
- `<think>` 推理已去净（v1 统一非思考工具调用）；所有 `tools` / `function_call` 均为合法 JSON

### 按来源

| 来源 | 保留 | License | 说明 |
|---|---|---|---|
| interstellarninja/hermes_reasoning_tool_use | 44,203 | Apache-2.0 | 主力，BFCL v3 对齐 |
| glaiveai/glaive-function-calling-v2 | 35,391 | Apache-2.0 | 铺量，**限 40k**（去重后 35k） |
| Team-ACE/ToolACE | 11,294 | Apache-2.0 | 多轮 / 拒答 |
| NousResearch/hermes-function-calling-v1 | 2,190 | Apache-2.0 | func-calling + singleturn |

### 按类别（对齐 BFCL）

| 类别 | 数量 | 含义 |
|---|---|---|
| multiturn | 29,158 | 多轮工具调用 |
| single | 20,270 | 单次调用 |
| no_call | 19,206 | 不该调工具（直接回答 / 拒答）|
| relevance | 14,742 | hermes_reasoning 自带的相关性标 |
| parallel | 5,365 | 一轮内并行多调用 |
| multistep | 4,337 | 多步顺序调用 |

> `no_call + relevance ≈ 34k`（36%）：专门训"该调才调"，对齐 BFCL 的 relevance/irrelevance。

### 丢弃原因

| 原因 | 数量 | 说明 |
|---|---|---|
| no_assistant | 2,180 | 无可学习的助手回应（多为去 think 后只剩纯推理无答案）|
| call_undefined_tool | 477 | 调用了未声明的工具（如隐式 python_interpreter），真·噪声 |
| tool_before_call | 191 | 工具返回出现在任何调用之前，顺序异常 |

## 关键设计

- 统一中间表示(IR) → LlamaFactory sharegpt+tools，详见 [`data-format.md`](data-format.md)。
- 各源真实格式坑均已处理：glaive 单引号嵌套 arguments / Hermes 说明文字里的空 `<tools>` 诱饵 / hermes_reasoning xLAM 扁平参数 + Python 类型名 / ToolACE Python 伪调用串 + 函数名含括号 + `type:dict`。
- 参数 schema 统一归一化为标准 JSON Schema（`type:dict`→`object`，`str`→`string` 等）。
