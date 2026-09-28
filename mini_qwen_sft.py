import os
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 可通过环境变量覆盖的运行参数（不设置时与原始脚本一致）
DATA_DIR = os.environ.get("DATA_DIR", "data")  # 快速体验可设为 mini_data
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 表示使用全部数据
NUM_PROC = int(os.environ.get("NUM_PROC", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "12"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
REPORT_TO = os.environ.get("REPORT_TO", "none")  # 需要wandb/tensorboard时改为对应值

# 加载分词器与模型
output_path = os.environ.get("OUTPUT_PATH", "results/sft")
model_path = os.environ.get("MODEL_PATH", "results/pt")
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def find_files(dirs):
    files = []
    for dir in dirs:
        base_path = os.path.join(DATA_DIR, "sft", dir)
        for dirpath, _, filenames in os.walk(top=base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files


# 加载数据集并进行预处理
directories = ["7M", "Gen"]
data_files = find_files(directories)
if not data_files:
    raise FileNotFoundError(f"在 {DATA_DIR}/sft 下没有找到parquet文件，请先运行 prepare.sh 下载数据")
print(f"微调数据文件数量: {len(data_files)}")
dataset = load_dataset(
    "parquet", data_files=data_files, split="train", columns=["conversations"]
)  # 只保留conversations字段
dataset = dataset.shuffle(seed=42)
if MAX_SAMPLES:
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
# print(dataset[:3]);input()


def preprocess_dataset(examples):
    """整理成 trl 的 prompt-completion 格式，loss 只作用在 completion(助手回复)上"""
    prompts, completions = [], []

    for conversations in examples["conversations"]:
        human_text, gpt_text = None, None
        for item in conversations:
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        # 与原始脚本保持一致：多轮对话只取最后一轮的问答
        # strip() 必需：回复首尾的空白会和模板换行合并成不同的token，
        # 导致 prompt 与 prompt+completion 的分词边界对不齐，completion掩码错位
        prompts.append(
            f"<|im_start|>user\n{human_text.strip()}<|im_end|>\n<|im_start|>assistant\n"
        )
        completions.append(f"{gpt_text.strip()}<|im_end|>")

    return {"prompt": prompts, "completion": completions}


# 应用预处理函数
train_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000,
    remove_columns=dataset.column_names,
    num_proc=NUM_PROC,
)
print(f"微调样本数: {len(train_dataset)}")

# 训练参数配置
training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    save_strategy="epoch",  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    # save_only_model=True,
    logging_steps=20,
    report_to=REPORT_TO,
    max_length=MAX_LENGTH,
    packing=False,
    completion_only_loss=True,  # 取代旧版的 DataCollatorForCompletionOnlyLM
    dataset_num_proc=NUM_PROC,
)

# 初始化Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    processing_class=tokenizer,
)

# 开始训练
trainer.train()
trainer.save_model()  # 保存模型
tokenizer.save_pretrained(output_path)  # 保存分词器


def plot_loss(save_directory, log_history):
    """绘制训练损失曲线并保存图像"""
    plt.switch_backend("agg")  # 使用非交互式后端
    key = "loss"  # 默认使用 'loss' 作为绘图的关键字
    steps, metrics = [], []

    # 提取损失数据
    for log_entry in log_history:
        if key in log_entry:
            steps.append(log_entry["step"])
            metrics.append(log_entry[key])

    # 绘制图像
    plt.figure()
    plt.plot(steps, metrics, color="#1f77b4", label="original")
    plt.title(f"Training {key} of {save_directory}")
    plt.xlabel("Step")
    plt.ylabel(key.capitalize())
    plt.legend()

    # 保存图像
    figure_path = os.path.join(save_directory, f"training_{key.replace('/', '_')}.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print(f"Figure saved at: {figure_path}")


# 绘制并保存损失曲线
plot_loss(output_path, trainer.state.log_history)
