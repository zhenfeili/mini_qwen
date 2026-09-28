#!/bin/bash
# 单卡在命令行运行
python mini_qwen_pt.py
python mini_qwen_sft.py
python mini_qwen_dpo.py

# 多卡在命令行运行
accelerate launch --config_file accelerate_config.yaml mini_qwen_pt.py
accelerate launch --config_file accelerate_config.yaml mini_qwen_sft.py
accelerate launch --config_file accelerate_config.yaml mini_qwen_dpo.py

# 多卡在后台运行
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_pt.py > output_pt.log 2>&1 &
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_sft.py > output_sft.log 2>&1 &
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_dpo.py > output_dpo.log 2>&1 &

du -sh */* | sort -h

du -sh * | sort -h

nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_pt1.py > output_pt1.log 2>&1 &
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_pt2.py > output_pt2.log 2>&1 &
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_pt3.py > output_pt3.log 2>&1 &
