# 统一数据格式规范

各数据源格式不一（glaive 是拼接字符串、hermes 系是 conversations + `<tool_call>` XML、ToolACE 又是另一套）。本项目先把它们统一成一个**内部中间表示（IR）**，再序列化成 **LlamaFactory sharegpt + tools** 格式喂训练。

## 1. 内部中间表示（IR）

每条样本：

```python
{
  "id": "glaive-000123",            # 来源-序号，便于追溯
  "source": "glaive",              # hermes_reasoning | toolace | hermes_v1 | glaive
  "category": "single",            # single | parallel | multistep | multiturn | relevance | irrelevant | unknown
  "system": "可选的 system 提示，不含工具定义（工具单列）",
  "tools": [                        # OpenAI 风格 function schema 列表
    {
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"]
      }
    }
  ],
  "turns": [                        # 按时间顺序的对话轮
    {"role": "user", "content": "北京天气怎么样"},
    {"role": "tool_calls", "calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
    {"role": "tool", "content": "{\"temp\": 25, \"desc\": \"晴\"}"},
    {"role": "assistant", "content": "北京今天 25℃，晴。"}
  ]
}
```

- `role` 取值：`user` | `assistant`（纯文本回答）| `tool_calls`（模型发起工具调用，可并行多个）| `tool`（工具返回结果）。
- **relevance / irrelevant（不该调工具）**样本：`turns` 里没有 `tool_calls`，直接 user → assistant，`tools` 仍提供——训练模型「该调才调」。

## 2. 目标格式：LlamaFactory sharegpt + tools

序列化规则（`scripts/build_dataset.py: to_llamafactory()`）：

| IR role | sharegpt `from` | `value` |
|---|---|---|
| `user` | `human` | 文本 |
| `assistant` | `gpt` | 文本 |
| `tool_calls`（1 个） | `function_call` | `{"name": ..., "arguments": {...}}` 的 JSON 字符串 |
| `tool_calls`（多个/并行） | `function_call` | `[{...}, {...}]` 的 JSON 数组字符串（⚠️ 需对齐 LlamaFactory 0.9.4 的并行调用解析，构建时校验）|
| `tool` | `observation` | 工具结果 JSON 字符串 |
| `system` | `system` | 文本（工具不放这里，由 LlamaFactory 模板按 `tools` 字段注入）|

输出样例（每行一条 JSONL）：

```json
{"conversations": [{"from": "human", "value": "北京天气怎么样"}, {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"}, {"from": "observation", "value": "{\"temp\": 25, \"desc\": \"晴\"}"}, {"from": "gpt", "value": "北京今天 25℃，晴。"}], "tools": "[{\"name\": \"get_weather\", \"description\": \"查询城市天气\", \"parameters\": {...}}]"}
```

对应 LlamaFactory `dataset_info.json`（`build_dataset.py` 会自动生成片段）：

```json
"fc_sft": {
  "file_name": "fc_sft_train.jsonl",
  "formatting": "sharegpt",
  "columns": {"messages": "conversations", "tools": "tools"},
  "tags": {
    "role_tag": "from", "content_tag": "value",
    "user_tag": "human", "assistant_tag": "gpt",
    "observation_tag": "observation", "function_tag": "function_call",
    "system_tag": "system"
  }
}
```

> Qwen3 原生用 Hermes 工具调用模板，LlamaFactory 训练时按 `tools` 字段 + 模型 chat template 自动渲染成 `<tools>...</tools>` / `<tool_call>...</tool_call>`，所以我们**不手动拼 XML**，只提供结构化 `tools` + 对话轮。

## 3. 清洗 / 校验规则（`validate_record`）

丢弃或修复不合规样本，构建时打印丢弃统计：

1. **结构**：至少 1 个 user 轮、1 个 assistant/tool_calls 轮；轮次角色合法。
2. **工具 schema**：`tools` 每项含 `name`/`parameters`，`parameters` 是合法 JSON Schema（至少 `type:object` + `properties`）。
3. **调用引用**：每个 `tool_calls.calls[].name` 必须在 `tools` 里定义（否则丢弃，避免教模型瞎调）。
4. **参数可解析**：`arguments` 能解析为 dict。
5. **tool 结果对齐**：有 `tool` 轮时其前面应有 `tool_calls`（顺序合理）。
6. **去重**：对 (system + tools + turns) 归一化后做 hash 去重。
7. **长度**：超 `--max-chars` 的样本计数（真实训练用 token 长度 recon 再定 `cutoff_len`）。

## 4. 切分

按 `source + category` 分层、固定 `--seed`，留出小比例（默认 1%，上限 N 条）作 dev 观察集；**正式能力评测以 BFCL 为准**，不靠 dev 集。
