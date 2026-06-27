# DPO 偏好对齐（修正 FC-SFT 的多步副作用）

> 用 LlamaFactory 对 SFT 后的 Qwen3-14B 做 **DPO 偏好对齐**，纠正窄域单轮 Function-Calling
> SFT 带来的一个副作用。配置见 `configs/qwen3_14b_lora_dpo.yaml`（精确超参以该文件为准），
> 数据构造脚本见 `scripts/build_dpo_dataset.py` / `scripts/validate_dpo_dataset.py`。

## 解决什么问题

窄域、单轮的 Function-Calling SFT 在提升「该不该调工具」判断的同时，会带来一个可观察的副作用：
在**多步**任务里，模型倾向于在拿到第一跳的部分结果后**直接编造**后续中间结果（如分布、路径、题号），
而不是继续真实调用下一个工具——表现为 `<think>` 被压短、第二跳早停。
纯靠推理时的编排层提示（reflect/nudge）只能缓解，属编排层上限；要从模型层根治，用 DPO 把
「该继续调工具」的偏好压回去。

> 这条副作用与对应的多步评测，在配套应用项目 [edu-agent](https://github.com/zhangxiaobina/edu-agent) 有完整记录。

## 偏好对从哪来

偏好信号来自**同一批多步任务、两个模型档位的轨迹对比**：

| 角色 | 来源 | 特征 |
|---|---|---|
| `chosen` | 强档 / 未微调档跑出的**成功**轨迹 | 每一跳都真实调用工具，`<think>` 完整 |
| `rejected` | SFT 档在**同一任务**上的**失败**轨迹 | 早停 / 编造中间结果、`<think>` 被压空 |

只保留「chosen 成功 且 rejected 失败」的任务——这些对才真正承载
『拿到部分数据后该继续调工具 vs 直接编造收尾』的偏好差。

> 仓库**不分发偏好数据**；提供的是脚手架与一条合成格式示例（`data/fixtures/fc_dpo_sample.jsonl`，
> 仅看结构，勿入训练集）。**数据够了即可照本文流程训练、可完整复现。** 轨迹 dump 由应用侧产生
> （edu-agent 提供 dump 脚本），或用任意能产出「同任务成功/失败轨迹对」的来源 / 公开数据。

## 数据流水线

```bash
# ① 配对：读两档轨迹 dump → 生成 LlamaFactory sharegpt-dpo 偏好对
#    （只保留 chosen 成功且 rejected 失败的任务）
python3 scripts/build_dpo_dataset.py \
    --chosen   traj_base.jsonl \
    --rejected traj_sft.jsonl \
    --tools    tools.openai.json \
    --out      data/processed/fc_dpo_train.jsonl \
    --mode     trajectory          # trajectory(默认,最稳) | turn(实验,信号更干净)

# ② 本地校验（纯标准库，无需 GPU / LlamaFactory，先挡格式坑省一轮 GPU 往返）
python3 scripts/validate_dpo_dataset.py data/processed/fc_dpo_train.jsonl --cutoff 16384
```

两种偏好范式：

- **trajectory**（默认）：整条轨迹折成一段 Hermes 文本作 chosen/rejected。最稳，LlamaFactory 原生支持。
- **turn**（实验性）：取两档工具调用序列的公共前缀作共享上文，只对「第一处分歧的那一步」做偏好。
  信号更干净（精准对准「该调工具 vs 编造」），但要先 `--do_train false` 预览一条，确认 template 对
  `conversations` 内 `function_call` / `observation` 多轮渲染正确。

## 在 `dataset_info.json` 注册

把下面 `fc_dpo` 条目并进 `data/processed/dataset_info.json`（与现有 SFT 数据集并列）。
`ranking: true` 让 LlamaFactory 按 chosen/rejected 处理；`columns` 比 SFT 多 `chosen` / `rejected` 两列：

```json
{
  "fc_dpo": {
    "file_name": "fc_dpo_train.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "tools": "tools"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "observation_tag": "observation",
      "function_tag": "function_call",
      "system_tag": "system"
    }
  }
}
```

## 训练

```bash
# 改 configs/qwen3_14b_lora_dpo.yaml 里的 /path/to/...（基座填 SFT-merged 完整权重目录）
llamafactory-cli train configs/qwen3_14b_lora_dpo.yaml
# 产出 DPO LoRA adapter → 复用 scripts/merge_lora.py 合并回基座
```

关键超参（完整见配置文件）：以 **SFT-merged 完整权重**为基座兼 reference、`stage: dpo`、
`pref_beta=0.1`、`lr=5e-7`（比 SFT 小两个量级）、`lora_rank=8`。

## 评测对照

合并后挂 vLLM，在 edu-agent 上跑多步任务子集（`scripts/eval_subset.py --cats multi_step`），
并回归一遍 BFCL V4，确认单轮 Function-Calling 能力没被 DPO 拖退化。建议对照四档：

| 档位 | 关注 |
|---|---|
| base（未微调） | 多步上限参照 |
| SFT | 副作用基线 |
| **SFT + DPO** | 多步是否回升（根治效果） |
| **SFT + DPO + W4A16** | 部署档是否保持 |

主看多步成功率、轨迹成功率、relevance/irrelevance（确认 DPO 没引入新的工具幻觉）；
工具精确率可能有代价，**如实记录**。

## 踩坑预警

- 基座必须是 `merge_and_unload` 后的 **SFT-merged 完整权重**，不能挂 adapter（同 SFT / 评测阶段的坑）。
- `dataset_info.json` 里 `ranking: true` + `columns` 要带 `chosen` / `rejected`，否则会被当普通 SFT 读。
- LlamaFactory DPO 字段是 `pref_beta` / `pref_loss` / `pref_ftx`（不是旧文档的 `dpo_beta`）。
- `pref_beta` 调参：0.1 起；多步无改善→0.05（更激进）；relevance/irrelevance 掉→0.2（更保守）。
- 训练时盯 `rewards/accuracies`（应 >0.5 且上行）、`rewards/margins`（应为正且增大）；不动多半是数据没承载偏好。
- DPO 后**必跑 BFCL 回归**——别为修多步把单轮 FC 的正向收益（抗工具幻觉）搞退化。
- 偏好数据少时易过拟合：先用少量对把链路（dump → 配对 → 校验 → 起训 → template 渲染）跑通，
  再扩充到足量正式训。质量 > 数量。
