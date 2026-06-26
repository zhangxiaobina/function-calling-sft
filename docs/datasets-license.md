# 数据集 License 复核清单

> 红线：本项目开源，只用 license 干净、可商用/可再分发训练数据的数据集；本仓库**不分发原始数据**，只提供下载 + 转换脚本。
> 真实学生数据/公司业务数据**绝不进**本仓库。

## ✅ 采用（全部 Apache-2.0，可开源）

| 数据集 | License | 规模 | 覆盖 | 用途 |
|---|---|---|---|---|
| `interstellarninja/hermes_reasoning_tool_use` | Apache-2.0 | ~51k | single / multistep / multiturn / relevance(no-call) / irrelevant，对齐 BFCL v3 | **主力** |
| `Team-ACE/ToolACE` | Apache-2.0 | ~11k | 多轮 / 拒答 | 补充 |
| `NousResearch/hermes-function-calling-v1` | Apache-2.0 | ~11.5k | Hermes XML / JSON-mode，与 Qwen 模板天然对齐 | 补充 |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | ~11万 | 单/多轮工具调用，量大 | 铺量 |

## ⚠️ 谨慎 / 需署名

| 数据集 | License | 说明 |
|---|---|---|
| `Salesforce/xlam-function-calling-60k` | **CC-BY-4.0**（非 NC）但 **gated + 需署名** | 可用但要走 gated 申请并在 README 署名；默认不纳入，需要再加 |

## ❌ 避开

| 数据集 | 原因 |
|---|---|
| `Salesforce/APIGen-MT-5k` | **CC-BY-NC**（非商用），不进开源训练集 |
| 所有 xLAM **模型权重** | research-only / NC |

## 复核动作（下载前逐个做）

1. 打开每个数据集的 HF 页面，确认 `license` 字段与上表一致（license 可能随版本变化）。
2. 记录确切的 commit / revision，写进 `build_dataset.py` 的 `REVISION`（可复现）。
3. gated 数据集（如 xlam）需先在 HF 同意条款。
4. 在最终 README 的「数据」一节列出所用数据集 + license + 链接（合规署名）。

## 评测数据

- **BFCL**：用 pip 包 `bfcl-eval`（Apache-2.0；别装裸 `bfcl`）。评测类别用 AST + relevance/irrelevance + multi-turn；**executable 类已退役不依赖**。
