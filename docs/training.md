# 训练（SFT LoRA）

> 用 LlamaFactory 对 Qwen3-14B 做 Function Calling LoRA SFT。实际产出 adapter 的配置见
> `configs/qwen3_14b_lora_sft_autodl.yaml`（精确超参以该文件为准）。

## 基座

- **`Qwen/Qwen3-14B`**（无 `-Instruct` 变体）。Qwen3 是思考/非思考混合模型；本项目 v1 用**非思考**数据训练
  （训练数据已去净 `<think>`），推理走 non-thinking 工具调用。

## 长度 recon（`scripts/recon_lengths.py`）

对训练集（含 tools 渲染）做 token 长度抽样统计：绝大多数样本远短于 4096，仅极少数接近上限。
→ 取 **`cutoff_len=4096`**：几乎零截断（截断会破坏工具调用序列、污染训练信号），14B LoRA 单卡可承受。
FC 数据普遍偏短，故配合 `packing + neat_packing` 把短序列打包填满 4096、消除 padding 浪费
（block-diagonal mask 保证打包的序列之间互不注意）。

## 方法与配置

- LoRA：rank / alpha / dropout、`target=all`（覆盖 q/k/v/o + gate/up/down 全线性层）——具体值见配置文件。
- 训练精度/加速：bf16 + SDPA + 梯度检查点；有效 batch 由 `per_device_batch × grad_accum` 控制。
- 现实路线：实际用**分层抽样子集 + 单 epoch**先出 before/after 信号（时间/成本约束）；
  全量数据 / 多 epoch 的等价配置见 `configs/qwen3_14b_lora_sft.yaml`（已附，未实跑）。

## 训练结果（定性）

- loss 健康收敛、无过拟合迹象。
- 产出 LoRA adapter（体积较小，不随仓库分发）。
- 下一步 = `scripts/merge_lora.py` 合并 → `scripts/run_bfcl.sh` 出 before/after（见 [`evaluation.md`](evaluation.md)）。

## 运行

```bash
llamafactory-cli train configs/qwen3_14b_lora_sft_autodl.yaml
```

## 关键提示（how-to-run）

- **`fa2 + neat_packing:true` 需每卡 `per_device_train_batch_size=1`**，否则 collator 会在 step0 断言失败；
  用 SDPA（非 fa2）时该约束不适用，可 batch>1。
- **flash-attn 在 aarch64 / 部分环境装不上**：退 `flash_attn: sdpa`（PyTorch 2.x 内置 SDPA 即走 FlashAttention 内核），无需单独编译。
- **载入 14B 的操作（训练/merge/量化）对内存/显存有实打实的要求**：在受限容器里需带卡运行，无卡模式只够装包/下载/写脚本。
