import argparse
import logging
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)

# 可选的模型路径：
#   results/pt  results/pt1  results/pt2  results/pt3   预训练模型（--base）
#   results/sft  results/sft/checkpoint-xxx             微调模型
#   results/dpo  results/dpo/checkpoint-xxx             直接偏好优化模型
parser = argparse.ArgumentParser(description="mini_qwen 推理/评估脚本")
parser.add_argument("--model_path", default="results/dpo", help="模型路径")
parser.add_argument(
    "--base",
    action="store_true",
    help="预训练模型（直接续写，不套用对话模板）",
)
parser.add_argument(
    "--prompt",
    action="append",
    default=None,
    help="非交互模式，可重复传入多个问题；不传则进入交互模式",
)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument(
    "--attn",
    default="flash_attention_2",
    help="注意力实现，显卡不支持时可设为 sdpa / eager",
)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    attn_implementation=args.attn,
    device_map="auto",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
print(f"已加载模型: {args.model_path}")


def chat(prompt):
    if args.base:
        text = prompt  # 预训练模型
    else:
        # 微调和直接偏好优化模型
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if args.prompt:  # 非交互模式
    for prompt in args.prompt:
        print("用户：", prompt)
        print("助手：", chat(prompt))
else:  # 交互模式
    while True:
        try:
            prompt = input("用户：")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if prompt.strip() in {"exit", "quit"}:
            break
        print("助手：", chat(prompt))
