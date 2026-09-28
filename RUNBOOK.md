# mini_qwen 单卡运行说明

本文档记录 2026-09-21 在 conda `base` 环境、单张 H200 上把
`mini_qwen_pt.py` / `mini_qwen_sft.py` / `mini_qwen_dpo.py` / `mini_qwen_eval.py`
四个脚本完整跑通的全过程：环境、数据准备、数据处理逻辑、训练配置、实际结果、复现命令，
以及为适配新版依赖所做的代码改动。

> 与 [README.md](README.md) 的分工：README 讲的是原作者在 6×H800 上用 16B token 训练的完整方案；
> 本文档讲的是**在当前这台单卡机器、当前这套依赖版本下如何把流程跑起来**，
> 并如实记录了因算力受限而缩减的训练规模。

## 目录

- [1. 环境](#1-环境)
- [2. 数据准备](#2-数据准备)
- [3. 数据处理逻辑](#3-数据处理逻辑)
- [4. 模型结构](#4-模型结构)
- [5. 模型训练](#5-模型训练)
- [6. 训练结果](#6-训练结果)
- [7. 推理与评估](#7-推理与评估)
- [8. 完整复现命令](#8-完整复现命令)
- [9. 代码适配说明](#9-代码适配说明)
- [10. 避坑清单](#10-避坑清单)
- [11. 已知限制](#11-已知限制)
- [12. .gitignore 说明](#12-gitignore-说明)

---

## 1. 环境

直接使用 conda `base` 环境，**没有安装任何新依赖，也没有降级任何库**。


| 项目           | 版本 / 规格                                             |
| ------------ | --------------------------------------------------- |
| Python       | 3.13.12（`/root/miniforge3`）                         |
| torch        | 2.10.0+cu128                                        |
| transformers | 4.57.6                                              |
| trl          | 1.2.0                                               |
| datasets     | 4.8.4                                               |
| accelerate   | 1.13.0                                              |
| flash_attn   | 2.8.3                                               |
| tokenizers   | 0.22.2                                              |
| peft         | 0.19.1                                              |
| modelscope   | 1.36.3                                              |
| matplotlib   | 3.11.2                                              |
| GPU          | 1 × NVIDIA H200 143GB（driver 570.86.15 / CUDA 12.8） |
| CPU / 内存     | 192 核 / 96GB                                        |


### 与 README 要求版本的差异

README 要求 `trl==0.11.4` + `transformers==4.45.0`，而 base 环境装的是
`trl==1.2.0` + `transformers==4.57.6`，跨了多个破坏性版本。

**这里选择改代码适配新版，而不是降级依赖**，原因是 base 环境被其它项目共用，
降级 transformers/trl 会牵连环境里其它推理与训练任务。具体改了什么见
[第 9 节](#9-代码适配说明)。

### 依赖自检

```bash
# 确认 flash-attn 真的能在这张卡上跑（装上不等于可用）
python -c "
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device='cuda', dtype=torch.bfloat16)
print('flash_attn OK:', flash_attn_func(q, q, q).shape)
"
```

### 关于多卡

`accelerate_config.yaml` 配的是 DeepSpeed ZeRO-2 + 6 卡，但**当前环境没有安装 deepspeed**，
且本机只有 1 张卡。因此本文档全程走 `run.sh` 里的"单卡在命令行运行"路径。
若要用多卡，需要先 `pip install deepspeed`，并把 `accelerate_config.yaml` 的
`num_processes` 改成实际 GPU 数。

---

## 2. 数据准备

三份数据都来自魔搭（ModelScope）上智源（BAAI）开源的数据集，用 `prepare.sh` 里的命令下载。
考虑到单卡算力，**只下载了子集**（原方案预训练数据 40G / 约 16B token）。

### 2.1 下载命令

```bash
cd /kpfs-llm-text/zhenfei.li/mini_qwen

# 预训练数据：只取"住宿餐饮酒店-中文-高质量"一个行业
modelscope download --dataset 'BAAI/IndustryCorpus2' \
    --include 'accommodation_catering_hotel/chinese/high*' --local_dir 'data/pt'

# 微调数据：Infinity-Instruct 的 7M 取前 3 个文件、Gen 取前 2 个文件
modelscope download --dataset 'BAAI/Infinity-Instruct' \
    --include '7M/train-0000[0-2]*' --local_dir 'data/sft'
modelscope download --dataset 'BAAI/Infinity-Instruct' \
    --include 'Gen/train-0000[0-1]*' --local_dir 'data/sft'

# 偏好数据：Infinity-Preference 全量（本身就不大）
modelscope download --dataset 'BAAI/Infinity-Preference' --local_dir 'data/dpo'
```

要下全量预训练数据就直接 `bash prepare.sh`（注意约 40G，且脚本里对 5 个大行业做了
"英文只留前 3 个文件"的裁剪）。

### 2.2 实际下载结果


| 文件                                                                     | 大小   | 行数      |
| ---------------------------------------------------------------------- | ---- | ------- |
| `data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet` | 371M | 94,201  |
| `data/sft/7M/train-00000-of-00075.parquet`                             | 78M  | 100,000 |
| `data/sft/7M/train-00001-of-00075.parquet`                             | 78M  | 100,000 |
| `data/sft/7M/train-00002-of-00075.parquet`                             | 78M  | 100,000 |
| `data/sft/Gen/train-00000-of-00015.parquet`                            | 187M | 100,000 |
| `data/sft/Gen/train-00001-of-00015.parquet`                            | 188M | 100,000 |
| `data/dpo/train-00000-of-00001.parquet`                                | 206M | 59,338  |
| `data/dpo/test-00000-of-00001.parquet`                                 | 350K | 100     |


合计约 1.15G：`data/pt` 371M、`data/sft` 576M、`data/dpo` 206M。

### 2.3 数据 schema

三个脚本对字段的假设如下，下载后可以用 `utils/demo_view_data.py` 或下面的命令核对：

```bash
python -c "
import pyarrow.parquet as pq
print(pq.ParquetFile('data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet').schema_arrow)
"
```

**预训练（IndustryCorpus2）**——脚本只读 `text` 字段，其余质量分列忽略：

```
text: string
alnum_ratio / avg_line_length / char_rep_ratio / flagged_words_ratio: double
max_line_length / num_words: int64
perplexity / quality_score / special_char_ratio / word_rep_ratio: double
_id: int64
industry_type: string
```

**微调（Infinity-Instruct）**——脚本只读 `conversations` 字段：

```
id: int64
conversations: list<struct<from: string, value: string>>
label / langdetect / source / reward: ...
```

`from` 字段实测只有 `human` 和 `gpt` 两种取值（抽样 2000 条统计：各 2455 次）。
轮数分布以 2 轮（单轮问答）为主：抽样 2000 条中 1849 条是 2 轮，其余为 4~30 轮的多轮对话。

**偏好（Infinity-Preference）**：

```
task_category: string
prompt: string
chosen:   list<struct<content: string, role: string>>
rejected: list<struct<content: string, role: string>>
id: int64
```

`chosen` / `rejected` 都是 2 元素列表，`[0]` 是 user、`[1]` 是 assistant，
脚本用 `assert` 校验了 `[1]["role"] == "assistant"`。

### 2.4 不下载数据也能跑

仓库自带 `mini_data/`（约 1.8M，由 `utils/save_mini_data.py` 生成），
三个训练脚本都支持 `DATA_DIR=mini_data` 环境变量，用于秒级验证流程是否通：

```bash
DATA_DIR=mini_data MAX_SAMPLES=100 BLOCK_SIZE=256 BATCH_SIZE=4 GRAD_ACCUM=2 \
    OUTPUT_PATH=/tmp/smoke_pt python -u mini_qwen_pt.py
```

---

## 3. 数据处理逻辑

三个阶段的数据处理方式完全不同，这一节说明各自在做什么、为什么这么做。

### 3.1 预训练：序列拼接 + 定长分块

`mini_qwen_pt.py` 的 `preprocess_dataset()`：

1. 每篇文档末尾追加结束符 `<|im_end|>`；
2. `tokenizer(..., add_special_tokens=False)` 分词；
3. 把整批的 token id **首尾相接拼成一条长序列**；
4. 按 `block_size=1024` 切成等长块，末尾不足一块的部分丢弃。

这是标准的 packing 做法：没有 padding，每个 token 都参与 loss，
GPU 利用率最高。代价是一篇文档可能被切断到两个 block、
一个 block 里可能混有两篇不相关的文档（靠 `<|im_end|>` 分隔）。

实际效果：**94,201 篇文档 → 132,802 个 block → 135,989,248 个 token（约 1.36 亿）**。

标签由 `DataCollatorForLanguageModeling(mlm=False)` 生成（即 `labels = input_ids`，
由模型内部做 shift），做的是纯自回归续写训练。

### 3.2 微调：prompt-completion + 只在回复上算 loss

原脚本用 `formatting_prompts_func` 拼出完整文本，再用
`DataCollatorForCompletionOnlyLM(response_template_ids)` 在 collator 里
找到 `<|im_start|>assistant\n` 的位置、把它之前的 label 全置为 -100。

**该 collator 在 trl 1.x 已被删除**，现改为 trl 原生的 prompt-completion 数据格式：

```python
prompt     = f"<|im_start|>user\n{human_text.strip()}<|im_end|>\n<|im_start|>assistant\n"
completion = f"{gpt_text.strip()}<|im_end|>"
```

配合 `SFTConfig(completion_only_loss=True)`，trl 会分别对 `prompt` 和
`prompt + completion` 分词，用两者长度差构造 `completion_mask`，
效果与原来的 collator 等价。

多轮对话的处理与原脚本保持一致：循环里不断覆盖 `human_text` / `gpt_text`，
因此**只保留最后一轮问答**。抽样统计里 92% 的样本本来就是单轮，影响有限。

#### `.strip()` 不是可选的

`gpt_text` 首尾的空白会和模板里的换行合并成**不同的 token**，
导致 `prompt` 的 token 序列不再是 `prompt + completion` 的前缀，
`completion_mask` 的边界随之错位（trl 只会打一条 warning 然后继续训）。

实测 2000 条样本中有 **127 条**触发该问题；加 `.strip()` 后归零。

#### 掩码正确性验证

改完之后实测校验过一条样本，确认 loss 只落在助手回复上：

```
完整序列:        '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！很高兴见到你。<|im_end|>'
参与loss的token: '你好！很高兴见到你。<|im_end|>'
被掩码的token:   '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
```

与 `DataCollatorForCompletionOnlyLM` 的语义完全一致。

注意 SFT 阶段 **不做 packing**（`packing=False`，与原项目一致），
每条样本独立 padding 到 batch 内最长长度。这是 SFT 吞吐明显低于 PT 的主要原因
（见 [6.4 吞吐对比](#64-吞吐与规模取舍)）。

### 3.3 直接偏好优化：标准偏好三元组

`mini_qwen_dpo.py` 把数据整理成 trl 的 standard preference 格式：

```python
prompt   = f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
chosen   = f"{chosen[1]['content'].strip()}<|im_end|>"
rejected = f"{rejected[1]['content'].strip()}<|im_end|>"
```

`.strip()` 的理由与 SFT 相同。DPOTrainer 在内部会自动复制一份冻结的
参考模型（reference model），因此显存占用约为 SFT 的两倍。

---

## 4. 模型结构

不使用现成权重，而是**以 Qwen2.5-0.5B-Instruct 的配置为骨架、扩大规模后随机初始化**。
仓库里 `models/Qwen2.5-0.5B-Instruct/` 只有 config 和 tokenizer 文件（没有权重），
这对预训练来说够了——`AutoModelForCausalLM.from_config()` 不需要权重。


| 配置项                   | Qwen2.5-0.5B-Instruct | mini_qwen | 说明   |
| --------------------- | --------------------- | --------- | ---- |
| `hidden_size`         | 896                   | **1024**  | 脚本覆盖 |
| `num_hidden_layers`   | 24                    | **48**    | 脚本覆盖 |
| `num_attention_heads` | 14                    | **16**    | 脚本覆盖 |
| `num_key_value_heads` | 2                     | **4**     | 脚本覆盖 |
| `intermediate_size`   | 4864                  | 4864      | 沿用   |
| `vocab_size`          | 151936                | 151936    | 沿用   |
| `tie_word_embeddings` | true                  | true      | 沿用   |


实际参数量 **998,810,624（约 1.00B）**，bf16 权重文件 1.99GB。

注意力实现为 `flash_attention_2`，计算精度 bf16。

> **坑**：transformers ≥ 4.54 会在 `AutoConfig.from_pretrained()` 时按**原始层数**
> 生成一个 `layer_types` 列表。脚本把 `num_hidden_layers` 从 24 改成 48 之后，
> 这个列表还是 24 项，建模时按层号索引会直接 `IndexError: list index out of range`。
> 因此改完层数必须同步重建：
>
> ```python
> if getattr(config, "layer_types", None) is not None:
>     config.layer_types = ["full_attention"] * config.num_hidden_layers
> ```
>
> 用 `full_attention` 是因为该 config 里 `use_sliding_window: false`，
> 原始 24 项也全是 `full_attention`。

---

## 5. 模型训练

### 5.1 环境变量开关

为了便于在单卡上调规模，三个训练脚本都加了环境变量覆盖，
**不设置时的默认值与原脚本完全一致**：


| 变量            | 默认值                    | 作用                              |
| ------------- | ---------------------- | ------------------------------- |
| `DATA_DIR`    | `data`                 | 数据根目录，设 `mini_data` 可秒级试跑       |
| `MAX_SAMPLES` | `0`                    | 限制样本数，`0` 表示全部                  |
| `NUM_PROC`    | `16`                   | `dataset.map` 并行进程数             |
| `BATCH_SIZE`  | pt/sft `12`、dpo `4`    | `per_device_train_batch_size`   |
| `GRAD_ACCUM`  | `16`                   | `gradient_accumulation_steps`   |
| `EPOCHS`      | pt `1`、sft/dpo `3`     | `num_train_epochs`              |
| `BLOCK_SIZE`  | `1024`                 | 仅 pt：分块长度                       |
| `MAX_LENGTH`  | `1024`                 | 仅 sft/dpo：截断长度                  |
| `MODEL_PATH`  | 见下                     | 输入模型路径                          |
| `OUTPUT_PATH` | `results/{pt,sft,dpo}` | 输出目录                            |
| `REPORT_TO`   | `none`                 | 上报后端，可设 `wandb` / `tensorboard` |


`MODEL_PATH` 默认值：pt 用 `models/Qwen2.5-0.5B-Instruct`（只读 config），
sft 用 `results/pt`，dpo 用 `results/sft`。

> 原 `mini_qwen_dpo.py` 里写的是 `results/sft/checkpoint-15461`（原作者 2 epoch 的产物），
> 这个步数与本次训练规模无关，已改为默认取 `results/sft`。
> 想从中间 checkpoint 接着训就显式传 `MODEL_PATH`。

### 5.2 本次实际使用的超参

三个阶段的学习率、warmup、调度器都沿用原脚本，只调整了 batch 切分
（原脚本是 6 卡各 12，这里单卡改成 24×8，等效 batch 仍为 192）：


|                  | pt                  | sft          | dpo           |
| ---------------- | ------------------- | ------------ | ------------- |
| 初始模型             | 随机初始化               | `results/pt` | `results/sft` |
| learning_rate    | 1e-4                | 1e-5         | 5e-7          |
| lr_scheduler     | cosine              | cosine       | cosine        |
| warmup_ratio     | 0.1                 | 0.1          | 0.1           |
| epochs           | 1                   | 3            | 3             |
| per_device_batch | 24                  | 24           | 8             |
| grad_accum       | 8                   | 8            | 8             |
| 等效 batch         | 192                 | 192          | 64            |
| 序列长度             | 1024                | 1024         | 1024          |
| 精度               | bf16                | bf16         | bf16          |
| 样本数              | 132,802 block       | 12,000 条     | 3,000 条       |
| 保存策略             | `save_steps=100000` | 每 epoch      | 每 epoch       |


### 5.3 启动命令

```bash
mkdir -p logs

TQDM_DISABLE=1 BATCH_SIZE=24 GRAD_ACCUM=8 NUM_PROC=32 \
    nohup python -u mini_qwen_pt.py > logs/run_pt.log 2>&1 &

# 等 pt 跑完后
TQDM_DISABLE=1 MAX_SAMPLES=12000 BATCH_SIZE=24 GRAD_ACCUM=8 NUM_PROC=32 \
    nohup python -u mini_qwen_sft.py > logs/run_sft.log 2>&1 &

# 等 sft 跑完后
TQDM_DISABLE=1 MAX_SAMPLES=3000 BATCH_SIZE=8 GRAD_ACCUM=8 NUM_PROC=32 \
    nohup python -u mini_qwen_dpo.py > logs/run_dpo.log 2>&1 &
```

`**python -u` 很重要**：不加 `-u` 时 stdout 是块缓冲的，
重定向到日志文件后可能半小时都看不到一行输出，很容易误判成卡死。
（第一次跑 pt 时就因为这个，30 分钟日志只有 3 行，得靠 `nvidia-smi` 采样
GPU 利用率才确认是在正常训练。）

`TQDM_DISABLE=1` 用来关掉进度条，否则日志里会塞满几千行 `\r` 刷新记录。

---

## 6. 训练结果

### 6.1 汇总


| 阶段  | 训练量                                             | 步数  | 耗时     | 吞吐         | loss                |
| --- | ----------------------------------------------- | --- | ------ | ---------- | ------------------- |
| pt  | 132,802 block × 1024 = **1.36 亿 token**，1 epoch | 692 | 76 min | 29.13 样本/s | 11.82 → **6.19**    |
| sft | 12,000 条 × 3 epoch，1,485 万 token                | 189 | 58 min | 10.34 样本/s | 9.53 → **8.87**     |
| dpo | 3,000 条 × 3 epoch，1,014 万 token                 | 141 | 38 min | 3.90 样本/s  | 0.6923 → **0.6928** |


流水线总耗时 **2 小时 52 分**（10,347 秒）。

### 6.2 loss 曲线

每个阶段结束时脚本会自动在输出目录画出 `training_loss.png`。

**pt**（每 20 步记录一次，下面每 5 条取 1 条）：

```
11.82 → 7.62 → 7.05 → 6.57 → 6.31 → 6.22 → 6.21
```

从随机初始化的 11.82（≈ ln(151936) = 11.93，即均匀分布）一路降到 6.19，
下降趋势健康，说明模型确实在学东西。

**sft**（`mean_token_accuracy` 为助手回复 token 的预测准确率）：


| step      | 20    | 40    | 60    | 80    | 100   | 120   | 140   | 160   | 180   |
| --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| loss      | 9.53  | 9.26  | 9.06  | 8.99  | 8.91  | 8.87  | 8.90  | 8.88  | 8.87  |
| token_acc | 0.051 | 0.066 | 0.076 | 0.080 | 0.082 | 0.086 | 0.086 | 0.086 | 0.087 |


**dpo**（`rewards/accuracies` 是 chosen 得分高于 rejected 的比例）：


| step       | 20     | 40     | 60     | 80     | 100    | 120    | 141    |
| ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| accuracies | 0.505  | 0.500  | 0.495  | 0.530  | 0.516  | 0.526  | 0.482  |
| margins    | 0.0020 | 0.0011 | 0.0011 | 0.0036 | 0.0027 | 0.0029 | 0.0033 |


DPO 的 loss 几乎不动（0.6923 → 0.6928，`ln2 ≈ 0.693`）、
`rewards/accuracies` 在 0.5 附近震荡，这是**预期结果**：
底座模型本身还在乱吐 token，对 chosen / rejected 两条回复没有任何偏好可言，
DPO 自然学不到有效的偏好信号。要让 DPO 有意义，前提是 SFT 模型已经能说通顺的话。

### 6.3 产物

```
results/
├── pt/                     7.5G
│   ├── model.safetensors           1.99G（1.00B 参数，bf16）
│   ├── config.json / generation_config.json
│   ├── tokenizer.json / vocab.json / merges.txt / ...
│   ├── training_loss.png
│   └── checkpoint-692/
├── sft/                     19G
│   ├── model.safetensors
│   ├── training_loss.png
│   └── checkpoint-63/ checkpoint-126/ checkpoint-189/    # 每 epoch 一个
└── dpo/                     19G
    ├── model.safetensors
    ├── training_loss.png
    └── checkpoint-47/ checkpoint-94/ checkpoint-141/     # 每 epoch 一个
```

合计约 **45G**。checkpoint 目录大是因为默认同时保存了优化器状态；
只想留权重可以打开脚本里注释掉的 `save_only_model=True`。

### 6.4 吞吐与规模取舍

sft 阶段**最初按 50,000 条起跑，实测约需 4.8 小时**，主动中止后降到 12,000 条重跑；
dpo 同理降到 3,000 条。这是为了当天跑完流水线做的取舍，不是脚本限制——
去掉 `MAX_SAMPLES` 就是全量。

三个阶段单样本吞吐差异很大，原因如下：


| 阶段  | 吞吐        | 原因                                                          |
| --- | --------- | ----------------------------------------------------------- |
| pt  | 29.1 样本/s | 序列拼接 + 定长分块，零 padding，每个 token 都参与计算                        |
| sft | 10.3 样本/s | `packing=False`，变长样本 padding 到 batch 内最长；实测 token 填充率仅约 40% |
| dpo | 3.9 样本/s  | 每步要跑 4 次前向（策略/参考模型 × chosen/rejected）                       |


全量跑的耗时估算（单卡 H200）：

- pt 全量 40G 数据（约 16B token）：按 1.36 亿 token / 76 min 折算，约 **6 天**（149 小时）——原项目用 6×H800 跑了 25 小时。
- sft 全量 500,000 条 × 3 epoch：约 **40 小时**。
- dpo 全量 59,438 条 × 3 epoch：约 **13 小时**。

想在单卡上提高 sft 吞吐，可以考虑开 `group_by_length=True` 减少 padding 浪费
（本次为了不改变训练语义没有启用）。

### 6.5 显存占用


| 阶段                        | 峰值显存   |
| ------------------------- | ------ |
| pt（batch 24 × 1024）       | 约 122G |
| sft（batch 24 × 1024）      | 约 126G |
| dpo（batch 8 × 1024，含参考模型） | 约 94G  |


显存主要不是被权重吃掉的（1B 模型 bf16 才 2G），而是
**词表 151936 带来的 logits 张量**：batch 24 × seq 1024 × vocab 151936 的
fp32 logits 单份就是 14.9G，加上 loss 计算中的副本和 48 层激活，很快就上百 G。

显存不够时优先调小 `BATCH_SIZE` 并同步调大 `GRAD_ACCUM`（保持等效 batch 不变），
例如 `BATCH_SIZE=12 GRAD_ACCUM=16`（即原脚本默认值）。
README 提到 pt/sft 12G、dpo 14G 显存即可训练，对应的是更小的 batch 和序列长度设置。

---

## 7. 推理与评估

`mini_qwen_eval.py` 已改为 argparse 驱动（改动原因见 [9.6](#96-eval-脚本改为-argparse-驱动)）。

### 参数


| 参数                 | 默认值                 | 说明                        |
| ------------------ | ------------------- | ------------------------- |
| `--model_path`     | `results/dpo`       | 模型路径                      |
| `--base`           | 关                   | 预训练模型模式：直接续写，不套对话模板       |
| `--prompt`         | 无                   | 非交互模式，可重复传入多个问题；不传则进入交互模式 |
| `--max_new_tokens` | 512                 | 生成长度                      |
| `--attn`           | `flash_attention_2` | 显卡不支持时可设 `sdpa` / `eager` |


### 用法

```bash
# 交互模式（原来的行为），输入 exit / quit 或 Ctrl-D 退出
python mini_qwen_eval.py --model_path results/dpo

# 非交互模式，适合脚本化批量测试
python mini_qwen_eval.py --model_path results/sft --prompt "介绍一下北京" --prompt "1+1=?"

# 预训练模型用 --base（续写模式，不套 <|im_start|> 模板）
python mini_qwen_eval.py --model_path results/pt --base --prompt "北京是中国的"
```

### 本次实际输出

```
########## results/pt --base ##########
用户： 这家酒店的早餐怎么样？
助手： ,在酒店的酒店,酒店,酒店,酒店,酒店,酒店,酒店,酒店,酒店,...

########## results/sft ##########
用户： 这家酒店的早餐怎么样？
助手：  - - - - - - - - - - - - - - - - - - - - - - - -

########## results/dpo ##########
用户： 这家酒店的早餐怎么样？
助手：  - - - - - - - - - - - - - - - - - - - - - - - -
```

**输出不通顺是训练量不足的必然结果**，不是脚本 bug：
原项目用 16B token 预训练，本次只有 1.36 亿 token，差了两个数量级。

有一个细节能反过来证明训练是真的生效了：pt 模型反复吐"酒店"这个词——
本次预训练语料正好只用了"住宿餐饮酒店"这一个行业，模型确实学到了该领域的高频词。
这也正是 README"复读机现象"一节讨论的现象在极端欠训练下的表现。

### 验证脚本本身是否正确

想确认 eval 脚本（模板拼接、生成、解码）没问题，可以拿一个真实的小模型跑：

```bash
python mini_qwen_eval.py --model_path /kpfs-llm-text/models/Qwen3-0.6B \
    --prompt "用一句话介绍北京" --max_new_tokens 120
```

实测输出通顺中文，说明脚本链路正确，问题只在于 mini_qwen 自身欠训练。
`/kpfs-llm-text/models/` 下 4B 以下的小 Qwen 模型还有
`Qwen3-1.7B-Instruct`、`Qwen3-4B`、`Qwen2.5-1.5B`、`Qwen3.5-2B` 等可选。

---

## 8. 完整复现命令

从零开始的完整流程，按顺序执行：

```bash
cd /kpfs-llm-text/zhenfei.li/mini_qwen
mkdir -p logs

# ---------- 0. 环境自检 ----------
python -c "
import torch, transformers, trl
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device='cuda', dtype=torch.bfloat16)
print('torch', torch.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__)
print('flash_attn OK:', flash_attn_func(q, q, q).shape)
"

# ---------- 1. 下载数据（约 1.15G，视网络 5~15 分钟） ----------
modelscope download --dataset 'BAAI/IndustryCorpus2' \
    --include 'accommodation_catering_hotel/chinese/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/Infinity-Instruct' \
    --include '7M/train-0000[0-2]*' --local_dir 'data/sft'
modelscope download --dataset 'BAAI/Infinity-Instruct' \
    --include 'Gen/train-0000[0-1]*' --local_dir 'data/sft'
modelscope download --dataset 'BAAI/Infinity-Preference' --local_dir 'data/dpo'

# ---------- 2. 先用 mini_data 验证流程（约 5 分钟） ----------
DATA_DIR=mini_data MAX_SAMPLES=200 BLOCK_SIZE=256 BATCH_SIZE=4 GRAD_ACCUM=2 NUM_PROC=4 \
    OUTPUT_PATH=/tmp/smoke_pt python -u mini_qwen_pt.py
DATA_DIR=mini_data MAX_SAMPLES=400 MAX_LENGTH=256 BATCH_SIZE=4 GRAD_ACCUM=2 NUM_PROC=4 EPOCHS=1 \
    MODEL_PATH=/tmp/smoke_pt OUTPUT_PATH=/tmp/smoke_sft python -u mini_qwen_sft.py
DATA_DIR=mini_data MAX_SAMPLES=200 MAX_LENGTH=256 BATCH_SIZE=2 GRAD_ACCUM=2 NUM_PROC=4 EPOCHS=1 \
    MODEL_PATH=/tmp/smoke_sft OUTPUT_PATH=/tmp/smoke_dpo python -u mini_qwen_dpo.py

# ---------- 3. 正式训练（合计约 2 小时 52 分） ----------
TQDM_DISABLE=1 BATCH_SIZE=24 GRAD_ACCUM=8 NUM_PROC=32 \
    python -u mini_qwen_pt.py 2>&1 | tee logs/run_pt.log

TQDM_DISABLE=1 MAX_SAMPLES=12000 BATCH_SIZE=24 GRAD_ACCUM=8 NUM_PROC=32 \
    python -u mini_qwen_sft.py 2>&1 | tee logs/run_sft.log

TQDM_DISABLE=1 MAX_SAMPLES=3000 BATCH_SIZE=8 GRAD_ACCUM=8 NUM_PROC=32 \
    python -u mini_qwen_dpo.py 2>&1 | tee logs/run_dpo.log

# ---------- 4. 评估 ----------
python mini_qwen_eval.py --model_path results/pt  --base --prompt "这家酒店的早餐怎么样？" --max_new_tokens 40
python mini_qwen_eval.py --model_path results/sft        --prompt "这家酒店的早餐怎么样？" --max_new_tokens 40
python mini_qwen_eval.py --model_path results/dpo        --prompt "这家酒店的早餐怎么样？" --max_new_tokens 40
```

想跑全量就去掉 `MAX_SAMPLES`，并把第 1 步换成 `bash prepare.sh`
（注意耗时，见 [6.4](#64-吞吐与规模取舍) 的估算）。

后台跑用 `nohup ... &` 替代 `| tee`，参考 [5.3](#53-启动命令)。

---

## 9. 代码适配说明

本次为跑通四个脚本所做的改动。前 5 项是**必须改否则直接报错或静默训错**的问题。

### 9.1 `config.layer_types` 与层数不同步（pt，直接崩）

transformers ≥ 4.54 在加载 config 时按原始 24 层生成 `layer_types` 列表，
脚本把 `num_hidden_layers` 改成 48 后该列表没有同步，建模按层号索引时报
`IndexError: list index out of range`。修复方式见 [第 4 节](#4-模型结构)。

### 9.2 `DataCollatorForCompletionOnlyLM` 已从 trl 删除（sft，直接崩）

trl 1.x 移除了这个 collator。改用 trl 原生的 prompt-completion 格式 +
`SFTConfig(completion_only_loss=True)`，并实测校验了掩码等价性。
详见 [3.2](#32-微调prompt-completion--只在回复上算-loss)。

### 9.3 分词边界错位导致掩码错误（sft/dpo，静默训错）

回复首尾空白与模板换行合并成不同 token，使 `completion_mask` 边界错位。
实测 2000 条中 127 条受影响。修复：对 `human_text` / `gpt_text` /
`chosen` / `rejected` / `prompt` 统一加 `.strip()`。

这一项最危险——trl 只打 warning 不报错，不注意的话会以错误的掩码训完整个流程。

### 9.4 DPOTrainer / DPOConfig 参数迁移（dpo，直接崩）


| 原写法                                 | trl 1.2 写法                           |
| ----------------------------------- | ------------------------------------ |
| `DPOTrainer(tokenizer=...)`         | `DPOTrainer(processing_class=...)`   |
| `DPOTrainer(max_length=...)`        | 移入 `DPOConfig(max_length=...)`       |
| `DPOTrainer(dataset_num_proc=...)`  | 移入 `DPOConfig(dataset_num_proc=...)` |
| `DPOTrainer(max_prompt_length=...)` | **已移除**，trl 1.2 不再支持                 |


SFTTrainer 同理：`max_seq_length` → `SFTConfig(max_length=...)`，
`packing` / `dataset_num_proc` 移入 config，`dataset_batch_size` 已移除，
`tokenizer=` → `processing_class=`。

### 9.5 wandb 自动检测导致训练无法启动（三个脚本，直接崩）

base 环境装了 wandb，`Trainer` 会自动检测并索要 API key，
没登录就抛 `wandb.errors.errors.UsageError: No API key configured`。
三个脚本都加了 `report_to=REPORT_TO`，默认 `none`；
要用 wandb 就 `REPORT_TO=wandb`（需先 `wandb login`）。

### 9.6 eval 脚本改为 argparse 驱动

原脚本有 11 行连续覆盖的 `model_path` 赋值（实际生效的只有最后一行
`results/dpo`），且是死循环 `input()`，无法脚本化调用、Ctrl-D 会抛
`EOFError` 栈。现改为 argparse，保留交互模式为默认行为，
另加 `--prompt` 非交互模式、`--base` 预训练模式、`--attn` 降级开关，
并补上 `pad_token_id` 和 `torch.no_grad()`。

### 9.7 其它小改动

- `torch_dtype=` → `dtype=`（transformers 4.56+ 重命名，旧名已废弃）。
- 数据目录找不到文件时抛 `FileNotFoundError` 并提示先跑 `prepare.sh`，
而不是让 `load_dataset` 抛一个难懂的空文件列表错误。
- 打印实际数据文件数和样本数，方便确认数据是否加载正确。
- 加入 [5.1](#51-环境变量开关) 的环境变量开关，默认值与原脚本一致。

### 9.8 一个可以忽略的告警

运行时会看到：

```
The tokenizer you are loading from 'results/pt' with an incorrect regex pattern:
... Mistral-Small-3.1-24B-Instruct-2503 ... You should set the `fix_mistral_regex=True` flag
```

这是 transformers 4.57 的**误报**。实测对比过原始 tokenizer 与保存后的 tokenizer，
对中英文、代码、特殊 token 各种输入的编码结果**逐 token 完全相同**：

```bash
python -c "
from transformers import AutoTokenizer
a = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct')
b = AutoTokenizer.from_pretrained('results/pt')
tests = ['你好，世界！\nHello world 123', 'def f(x):\n    return x\n\n# done',
         '<|im_start|>user\n测试<|im_end|>\n<|im_start|>assistant\n']
print('identical:', all(a(t).input_ids == b(t).input_ids for t in tests))
"
```

可以放心忽略。

---

## 10. 避坑清单

1. **一定要用 `python -u`**。不加的话 stdout 块缓冲，`nohup` 日志可能几十分钟没有一行，
  极易误判成卡死。判断是否真的在训练可以看 GPU 利用率采样：
2. `**TQDM_DISABLE=1**`，否则日志被进度条刷新记录塞满。
3. `**.strip()` 不能省**。见 [9.3](#93-分词边界错位导致掩码错误sftdpo静默训错)，
  这是唯一一个不报错但会训错的问题。
4. **改 `num_hidden_layers` 必须同步改 `layer_types`**。见 [9.1](#91-configlayer_types-与层数不同步pt直接崩)。
5. **显存看 logits 不看权重**。1B 模型权重才 2G，但词表 151936 让 logits 张量成为显存大头。
  OOM 时调小 `BATCH_SIZE`、同步调大 `GRAD_ACCUM` 保持等效 batch。
6. **大文件不要提交进 git**。本次产生了约 45G 产物 + 1.15G 数据，
  已加 `[.gitignore](.gitignore)` 统一忽略，详见 [第 12 节](#12-gitignore-说明)。
7. **多卡需要先装 deepspeed**。`accelerate_config.yaml` 配的是 ZeRO-2 + 6 卡，
  但当前环境没装 deepspeed，且 `num_processes` 需按实际卡数改。
8. **先用 `mini_data` 跑通再上真数据**。三个脚本都支持 `DATA_DIR=mini_data`，
  5 分钟能验证完整条链路，比直接上全量数据等一小时才发现报错划算得多。

## 11. 已知限制

- **训练规模远小于原方案**：pt 1.36 亿 token（原方案 16B）、sft 12,000 条（原方案 9M）、
dpo 3,000 条（原方案 60K）。因此模型输出不通顺，**本次目标是跑通流程而非复现效果**。
- **预训练语料只有一个行业**（住宿餐饮酒店/中文/高质量），领域严重偏窄，
模型学到的高频词集中在该领域。
- **多轮对话被截断成单轮**：沿用原脚本逻辑，`conversations` 只保留最后一轮问答。
抽样统计 92% 的样本本来就是单轮，但多轮样本的前文信息被丢弃了。
- **DPO 阶段没有学到有效偏好信号**：`rewards/accuracies` 在 0.5 附近震荡，
因为 SFT 底座还不能生成通顺文本。要让 DPO 有意义，必须先把 pt/sft 训到位。
- `**demo/demo_pt.py`、`demo_sft.py`、`demo_dpo.py` 未适配**，
它们与主脚本有同样的旧 API 问题（本次只处理了四个 `mini_qwen_*.py`）。
- `**accelerate_config.yaml` 未验证**：本次全程单卡，多卡 + DeepSpeed 路径没有实际跑过。

---

## 12. .gitignore 说明

仓库原先没有 `.gitignore`，而跑完一轮流水线会产生约 **45G 训练产物 + 1.15G 数据集**，
稍不注意就会被 `git add .` 带进版本库。现已添加 `[.gitignore](.gitignore)`，
原则是：**大模型权重、大数据集、训练产物一律不入库，仓库只保留代码、配置、文档，
以及 `mini_data/` 里用于跑通流程的小样本数据。**

### 忽略范围


| 类别           | 规则                                                                                                      | 本次实际命中                             |
| ------------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 训练产物         | `results/`                                                                                              | 45G（10 份权重各 1.9G + 7 份优化器状态各 3.8G） |
| 下载数据         | `data/`                                                                                                 | 1.15G（单文件 78M~371M）                |
| 训练日志         | `logs/`、`output_*.log`、`nohup.out`                                                                      | —                                  |
| 模型权重（兜底）     | `*.safetensors` `*.bin` `*.pt` `*.pth` `*.ckpt` `*.msgpack` `*.h5` `*.onnx` `*.gguf` `*.engine` `*.pkl` | —                                  |
| 大数据文件        | `*.parquet` `*.arrow` `*.npy` `*.npz` `*.tfrecord` `*.jsonl`                                            | —                                  |
| 压缩包          | `*.tar` `*.tar.gz` `*.tgz` `*.zip` `*.7z`                                                               | —                                  |
| 实验跟踪         | `wandb/`、`runs/`、`events.out.tfevents.`*                                                                | —                                  |
| Python / 编辑器 | `__pycache__/`、`*.py[cod]`、`.venv/`、`.vscode/`、`.idea/`、`.DS_Store`                                     | —                                  |


目录规则（`results/`、`data/`）已经覆盖了本次全部 37 个超过 10M 的文件；
后面那些按扩展名的规则是兜底，防止以后在别的目录里手滑提交权重或数据集。

### 一个必须保留的例外

`mini_data/` 下有 6 个已入库的小样本 parquet（100K~425K），是"不下载数据也能跑通流程"的
依赖项，会被 `*.parquet` 规则误伤，因此加了反向规则显式保留：

```gitignore
*.parquet
!mini_data/**/*.parquet
```

### 校验结果

添加后做了三项校验，均通过：

```bash
# 1. 所有 >10M 的文件都已被忽略（无输出即通过）
find . -path ./.git -prune -o -type f -size +10M -print | while read f; do
    git check-ignore -q "$f" || echo "未忽略: $f"
done

# 2. 原有 49 个已跟踪文件无一被误伤（无输出即通过）
git ls-files | while read f; do git check-ignore -q "$f" && echo "被误伤: $f"; done

# 3. git status 干净，只剩代码改动和新增文档
git status --short
```

第 3 项的实际输出：

```
 M mini_qwen_dpo.py
 M mini_qwen_eval.py
 M mini_qwen_pt.py
 M mini_qwen_sft.py
?? .gitignore
?? RUNBOOK.md
```

> 注意：`.gitignore` 只对**未跟踪**的文件生效。如果某个大文件已经被 `git add` 过，
> 需要先 `git rm --cached <file>` 把它从索引里移除，规则才会起作用。
> 本次检查确认没有已跟踪的文件超过 10M（最大的是 `models/Qwen2.5-0.5B-Instruct/tokenizer.json`，6.8M），
> 因此不需要做这一步。

