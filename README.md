# function-calling-sft

> 把开源基座 **Qwen3-14B** 后训练成一个更可控的**工具调用（Function Calling）引擎**：
> Hermes 数据工程 → LoRA SFT → 官方 **BFCL V4** 同机 before/after 评测 → **GPTQ W4A16 量化 + vLLM 部署**。
> 全流程可复现、数据 license 干净。

产出的量化模型可作为另一个项目 **[edu-agent](https://github.com/zhangxiaobina/edu-agent)**（多工具教学 Agent）的工具调用引擎——
一个仓库演示*工具调用模型的微调 / 评测 / 压缩部署*，另一个演示*把它搭成真实场景的多工具应用*。

---

## 解决什么问题

工具调用是 AI 应用 / Agent 的核心能力。本项目用一条干净的线，端到端走通**微调 → 评测 → 压缩部署**：

```
数据工程(Hermes→LlamaFactory)
        │  公开数据(Apache-2.0)清洗去重 → 统一工具调用训练集
        ▼
baseline BFCL V4 (微调前)  ──┐
        │                    │ 同机、同 vLLM、八类、各独立 result-dir
   LoRA SFT (LlamaFactory)   │ → 公平 before/after
        │                    │
   merge_and_unload          │
        ▼                    │
   BFCL V4 (微调后) ─────────┘
        ▼
   GPTQ W4A16 量化 (llm-compressor, in-domain 校准)
        ▼
   vLLM(Marlin) 部署 → BFCL 复评 + 解码吞吐 bench
        ▼
   作为 edu-agent 的工具调用引擎(本地单卡 vLLM 端点)
```

## 主要结论（定性）

> 本项目刻意在一个**很强的基座**上做工具调用 SFT，因此结论比「分数大涨」更值得说清楚：

- **抗工具幻觉提升**：微调最稳健的真实增益是 **Irrelevance（该不该调工具的判断）明显变好**——更少「不该调却调」的工具幻觉。
- **整体基本持平**：强基座 FC 本就接近天花板，一份通用数据 SFT 后**整体 BFCL 准确率基本持平**；
  逐条看判错案例后确认，剩余差异主体是 **BFCL 标准答案的字符串约定**（如地名是否翻译/补州名），而非模型能力——
  刻意去刷这类分属于过拟合评测，不做。
- **W4A16 量化高性价比**：体积与运行时显存压到**约 1/3**，BFCL 八类整体**基本无损**，单流解码**约 3× 提速**，使 14B 可单卡本地部署。
- **量化「无损」有边界**：上面的「基本无损」仅在 BFCL **单轮 AST 口径**成立；接进多步 agentic 任务时，
  量化叠加窄域 SFT 会在长链路上放大误差——这条反直觉但真实的结论在 [edu-agent](https://github.com/zhangxiaobina/edu-agent) 有完整记录。
- **DPO 修正多步副作用**：窄域单轮 FC-SFT 会让模型在多步任务里倾向「编造」中间结果而非续调工具；
  本仓附一条 **DPO 偏好对齐**链路（`base 成功轨迹` vs `SFT 编造轨迹` 的偏好对）从模型层把「该调工具」压回去，
  脚手架与复现见 [`docs/dpo.md`](docs/dpo.md)。

> 复现后能在本地拿到逐类的精确数字（脚本与口径见下文）；本 README 只给定性结论，不替读者的硬件/版本背书具体数值。

## 架构与文件树

```
function-calling-sft/
├── README.md                         本文件
├── LICENSE                           Apache-2.0
├── requirements.txt                  数据下载/转换/长度recon 依赖(pipeline 核心仅用 stdlib)
├── configs/                          LlamaFactory 训练配置
│   ├── qwen3_14b_lora_sft_autodl.yaml  实际产出 adapter 的配置(单卡,子集,1ep,packing)
│   ├── qwen3_14b_lora_sft.yaml         全量数据/多 epoch 的等价配置
│   ├── qwen3_14b_lora_dpo.yaml         DPO 偏好对齐配置(stage:dpo, SFT-merged 作基座+reference)
│   ├── qwen3_14b_lora_probe.yaml       步速计时探针(只跑少量步,不产权重)
│   └── qwen3_14b_lora_probe2.yaml      步速探针 v2
├── data/
│   ├── fixtures/                     离线自测用的小样本(每源数条;入库)
│   │   └── fc_dpo_sample.jsonl         DPO 偏好对的合成格式示例(仅看结构,勿入训练集)
│   └── processed/                    构建产物(大 jsonl,.gitignore 不入库)
├── docs/
│   ├── data-format.md                统一 IR → LlamaFactory sharegpt+tools 规范
│   ├── dataset-stats.md              训练集构建统计(来源/类别/丢弃)
│   ├── datasets-license.md           数据源 license 复核清单
│   ├── training.md                   SFT 超参 + 长度 recon + 训练说明
│   ├── evaluation.md                 BFCL V4 评测方法学与复现
│   ├── quantization.md               GPTQ W4A16 量化方法学与复现
│   └── dpo.md                        DPO 偏好对齐:修正 FC-SFT 的多步副作用(方法+复现)
└── scripts/
    ├── inspect_dataset.py            看各数据源的真实原始字段
    ├── adapters.py                   各源(glaive/hermes/toolace) → 统一 IR 的适配器
    ├── build_dataset.py              下载+转换+清洗+校验+去重+分层切分 → 训练集
    ├── recon_lengths.py              Qwen3 tokenizer 长度 recon → 定 cutoff_len
    ├── subsample.py                  分层抽样出更小子集
    ├── test_pipeline.py              离线 pipeline 自测(用 fixtures,不联网)
    ├── merge_lora.py                 LoRA adapter → 合并回基座
    ├── run_bfcl.sh                   BFCL V4 generate + evaluate(八类)
    ├── quantize_w4a16.py             GPTQ W4A16 量化(in-domain 校准)
    ├── bench_decode.py               解码吞吐基准(bf16 vs W4A16)
    ├── build_dpo_dataset.py          两档轨迹 dump → 配偏好对(base成功/SFT失败) → sharegpt-dpo
    └── validate_dpo_dataset.py       本地校验 DPO 数据集合规(无需 GPU,挡格式坑)
```

## 研究思路与关键取舍

- **数据工程对齐基座原生模板**：多个 Apache-2.0 公开源各有格式坑，先统一成一套内部中间表示（IR），再序列化成
  **LlamaFactory sharegpt + tools**。不手拼 `<tool_call>` XML——交给 Qwen3 原生 Hermes 模板按 `tools` 字段渲染，保证训练/推理同构。
  专门保留一大批「不该调工具」的负样本来训练「该调才调」。详见 [`docs/data-format.md`](docs/data-format.md)、[`docs/dataset-stats.md`](docs/dataset-stats.md)。
- **为什么 merge 而不是在线挂 LoRA 评测**：BFCL 的 handler 始终按 base 路径发请求，在线挂 LoRA 不会真正路由到 adapter，
  会得到 after==before 的假结果。最稳的是 `merge_and_unload` 成独立目录，base/merged 各用 `--local-model-path` 平等评测。
- **为什么看 BFCL 分类明细、不看 overall**：BFCL V4 的 overall 把未跑的类别计入分母会假性压低，只看实际跑的八类的 correct/total。
- **强基座微调的收益边界**：base 本就很强、天花板低，通用数据 SFT 多为持平略降；唯一稳健正向是抗工具幻觉。
  能界定这条边界、看懂 BFCL 判错机制，比追一个虚高的「全面提升」更有意义。详见 [`docs/evaluation.md`](docs/evaluation.md)。
- **W4A16 而非 FP4 / W8A8**：解码主要带宽受限，权重压到 4-bit 直接转吞吐；激活保留 16-bit → AST 精度损失极小。
  用 **in-domain 的 FC 数据**做 GPTQ 校准，比通用语料更贴工具调用的激活分布。详见 [`docs/quantization.md`](docs/quantization.md)。

## 完整复现步骤

> 数据工程脚本仅依赖标准库即可离线自测；真实下载 / 训练 / 评测 / 量化需对应环境（见各步注释）。

```bash
# ── 0) 离线自测：不联网，用 data/fixtures 小样本跑通整条 pipeline 逻辑 ──
python3 scripts/test_pipeline.py

# ── 1) 数据工程：下载 + 转换 + 清洗 + 去重 + 分层切分 → 统一训练集 ──
pip install -r requirements.txt
python3 scripts/inspect_dataset.py --source glaive --limit 3          # 先看真实字段
python3 scripts/build_dataset.py --out data/processed \
        --sources hermes_reasoning toolace hermes_v1 glaive           # 产出 fc_sft_train.jsonl(+dev)
python3 scripts/recon_lengths.py                                      # 长度 recon → 印证 cutoff_len
python3 scripts/subsample.py --in data/processed/fc_sft_train.jsonl \
        --out data/processed/fc_sft_train_35k.jsonl                  # (可选) 分层抽子集

# ── 2) SFT LoRA（GPU 环境，LlamaFactory）──
llamafactory-cli train configs/qwen3_14b_lora_sft_autodl.yaml

# ── 3) 合并 adapter 回基座（独立全权重目录，供 BFCL 平等评测）──
python scripts/merge_lora.py \
    --base /path/to/Qwen3-14B --adapter /path/to/qwen3-14b-fc-lora-adapter \
    --out  /path/to/Qwen3-14B-fc-merged

# ── 4) BFCL V4 同机评测：base / merged 各跑一次出 before/after ──
scripts/run_bfcl.sh /path/to/Qwen3-14B           base
scripts/run_bfcl.sh /path/to/Qwen3-14B-fc-merged merged

# ── 5) GPTQ W4A16 量化（in-domain 校准）+ 复评 + 吞吐 bench ──
python scripts/quantize_w4a16.py \
    --model /path/to/Qwen3-14B-fc-merged \
    --out   /path/to/Qwen3-14B-fc-merged-W4A16 \
    --calib data/processed/fc_sft_train.jsonl
scripts/run_bfcl.sh /path/to/Qwen3-14B-fc-merged-W4A16 w4a16          # 验量化是否无损
python scripts/bench_decode.py /path/to/Qwen3-14B-fc-merged       merged_bf16
python scripts/bench_decode.py /path/to/Qwen3-14B-fc-merged-W4A16 w4a16

# ── 6) (可选) DPO 偏好对齐：修正窄域 SFT 的多步「编造」副作用 ──
#     偏好对由两档轨迹 dump 配对生成(chosen=成功轨迹/rejected=同任务编造轨迹)；仓库不分发数据。
python3 scripts/build_dpo_dataset.py --chosen traj_base.jsonl --rejected traj_sft.jsonl \
        --tools tools.openai.json --out data/processed/fc_dpo_train.jsonl --mode trajectory
python3 scripts/validate_dpo_dataset.py data/processed/fc_dpo_train.jsonl --cutoff 16384
# 在 data/processed/dataset_info.json 注册 fc_dpo(见 docs/dpo.md) 后训练：
llamafactory-cli train configs/qwen3_14b_lora_dpo.yaml                # 详见 docs/dpo.md
```

> 模型权重（LoRA adapter / 量化模型）体积较大，不随仓库分发；按上面步骤可自行产出，后续可能另行发布。

## 局限与边界（诚实）

- 本项目重点是**走通并讲清整条链路**，而非在强基座上刷高 BFCL 分；通用数据 SFT 的整体增益有限，主要收益在抗工具幻觉。
- 本轮只跑八类，未覆盖 multi_turn；分数一律以分类明细为准，不用 BFCL overall。
- 训练用的是子集 / 单 epoch 的现实路线（时间/成本约束），全量配置已附但未实跑。
- W4A16 的「基本无损」仅在 BFCL 单轮 AST 口径成立；多步 agentic 场景的额外损伤见 [edu-agent](https://github.com/zhangxiaobina/edu-agent)。
- 异构硬件（如 GB10 / aarch64 统一内存）跑量化可能需要给显存查询日志打补丁，见 [`docs/quantization.md`](docs/quantization.md)。

## 数据 license

全部采用 **Apache-2.0** 的公开数据集，本仓库**只提供下载 + 转换脚本，不分发原始数据**；任何真实业务数据与密钥**绝不进库**。
逐源 license 见 [`docs/datasets-license.md`](docs/datasets-license.md)。

## License

代码 **Apache-2.0**（见 [`LICENSE`](LICENSE)）。各数据集 license 归其原作者。
