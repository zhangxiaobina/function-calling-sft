# BFCL V4 评测 · 方法学与复现

> before/after 硬信号怎么来的、怎么读。base 与 merged(微调后) 在**同一台机器、同一个 vLLM** 上跑全八类，
> 各给独立 `--result-dir`，保证公平对比。脚本 `scripts/run_bfcl.sh`、`scripts/merge_lora.py`。

## 1. 为什么用 BFCL，用哪几类

- **BFCL（Berkeley Function-Calling Leaderboard）** 是工具调用最权威的公开评测；用官方 pip 包 `bfcl-eval`（Apache-2.0），后端 vLLM 本地推理。
- 本轮跑**八类**：AST 四类 `simple_python / multiple / parallel / parallel_multiple` +
  live 三类 `live_simple / live_multiple / live_parallel` + `irrelevance`。
- **AST 口径**：把模型输出解析成抽象语法树，逐个比对函数名 + 参数键值是否落在「可接受答案集合」里。
- **relevance / irrelevance 口径**：irrelevance 是「给了工具但不该调」的题，考的是**抗工具幻觉**——正确行为是不调用、直接回答。

## 2. 三个会让 before/after 失真的坑（必须规避）

1. **LoRA 在线挂载评测会白跑**：BFCL 的 handler 始终用 base 路径发请求，`--enable-lora` 不会把请求路由到 adapter → after==before。
   **正解 = 先 `merge_and_unload`**（`scripts/merge_lora.py`），base/merged 各用 `--local-model-path` 平等评测。
2. **模型名必须带 `Qwen/` 前缀**：否则会被路由到云端 API。base 与 merged 都用同一个名，靠 `--local-model-path` 区分实际权重。
3. **别看 overall，看分类明细**：BFCL V4 的 overall 把没跑的类别计入分母会假性压低，结论以八类各自的 correct/total 为准。

## 3. 主要结论（定性）

- **唯一稳健正向是 Irrelevance（抗工具幻觉）明显提升**：微调后更会判断「不该调工具时就别调」。
- **整体基本持平**：base Qwen3-14B 的 FC 本就很强（AST 类 90%+ 量级、live 类 80% 量级），天花板低；
  一份通用 FC 数据 SFT → 大部分类别持平略降，是「微调强基座」的固有困境。
- **判错主体是评测答案的字符串约定，不是能力差**：逐条看 `score.json` 后确认，剩余判错多为标准答案的命名规范
  （地名是否翻译、是否补州名/限定词等，错误类型主体为 `value_error:string`）。去刷这类分 = 过拟合 BFCL，故不做。

> 复现后可在 `scores_<tag>/Qwen_Qwen3-14B-FC/{non_live,live}/BFCL_v4_<cat>_score.json` 看到逐类的精确 correct/total。
> 本文档只给定性结论，不替读者的硬件/版本背书具体数值。

## 4. 复现

```bash
scripts/run_bfcl.sh /path/to/Qwen3-14B           base
scripts/run_bfcl.sh /path/to/Qwen3-14B-fc-merged merged
# 逐类分数: scores_<tag>/Qwen_Qwen3-14B-FC/{non_live,live}/BFCL_v4_<cat>_score.json
# 逐条 case(含 possible_answer): 同名 *_score.json 的后续每行
```

量化模型的同口径复评见 [`quantization.md`](quantization.md)。
