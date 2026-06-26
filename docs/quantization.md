# GPTQ W4A16 量化 · 方法学与复现

> 把 merged 的 14B bf16 模型压成 4-bit 权重 / 16-bit 激活的 W4A16，单卡可部署，
> 接进 vLLM(Marlin) 跑 BFCL 复评 + 解码吞吐 bench。脚本 `scripts/quantize_w4a16.py`、`scripts/bench_decode.py`。

## 1. 为什么选 W4A16 + in-domain 校准

- **W4A16 而非 FP4 / W8A8**：14B 单流解码主要是**带宽受限**——权重从 16-bit 压到 4-bit 直接转成吞吐；
  激活保留 16-bit，AST 精度损失极小。这是「既要小又要不掉点」的稳妥点。
- **in-domain 校准**：GPTQ 用项目自己的工具调用训练数据做校准，比通用语料更贴 FC 场景的激活统计，量化掉点更小。
  校准样本按 chat template + `tools` 字段渲染（与训练/推理同构）。
- recipe 用 `GPTQModifier(targets=Linear, ignore=[lm_head], scheme=W4A16)`：只量化线性层、保留输出头精度。

## 2. 主要结论（定性）

- **体积与运行时显存压到约 1/3**（4-bit 权重）。
- **精度基本无损**：BFCL 八类整体准确率与量化前持平，单类别只有 ±数题的小波动、互相抵消。
- **解码约 3× 提速**（单流 / 低 batch）；高 batch 下加速收窄。

**为什么 INT4 拿不到「4× 加速」**：低 batch 是带宽受限，权重压缩比几乎线性给到吞吐；
batch 大了转 compute 受限，Marlin 把 4-bit 反量化回 fp16 的算力成本显现，加速收窄。这条曲线本身就是量化的体检。

> 复现后可在本地环境拿到逐项的精确数字（体积 / 显存 / 逐类题数 / 各 batch 的 tok/s）。本文档只给定性结论。

## 3. 异构硬件提示

- 在统一内存架构（如 NVIDIA GB10 / aarch64）上，量化工具的「逐层打印显存占用」可能调用不被支持的显存查询接口而报错，
  需要给该日志打补丁（用 try/except 兜住）才能跑通；这只是日志、不影响量化本身。
- `llm-compressor` 与 `transformers` 版本需匹配，否则 import 失败——以 `scripts/quantize_w4a16.py` 注释 / llm-compressor 官方说明为准。

## 4. 复现

```bash
# 量化（带 GPU 的环境）
python scripts/quantize_w4a16.py \
    --model /path/to/Qwen3-14B-fc-merged \
    --out   /path/to/Qwen3-14B-fc-merged-W4A16 \
    --calib data/processed/fc_sft_train.jsonl

# 复评是否无损（vLLM 自动认 compressed-tensors W4A16 → MarlinLinearKernel）
scripts/run_bfcl.sh /path/to/Qwen3-14B-fc-merged-W4A16 w4a16

# 吞吐 bench
python scripts/bench_decode.py /path/to/Qwen3-14B-fc-merged       merged_bf16
python scripts/bench_decode.py /path/to/Qwen3-14B-fc-merged-W4A16 w4a16
```

## 5. 部署接入

W4A16 模型用 vLLM 起 OpenAI 兼容端点（开 `--enable-auto-tool-choice --tool-call-parser hermes`），
即可作为 [edu-agent](https://github.com/zhangxiaobina/edu-agent) 多工具 Agent 的工具调用引擎（本地单卡）。
那边有把这个量化模型接进真实多步任务后的 agentic 对照与边界分析。
