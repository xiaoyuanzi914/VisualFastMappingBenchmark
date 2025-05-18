#!/bin/bash
model_name="doubao-1.5-vision-pro-32k-250115"
# model chocie 
## "doubao-1.5-vision-pro-32k-250115" 
## "Qwen*Qwen2.5-VL-72B-Instruct" 
## "deepseek-ai*deepseek-vl2" 
## "gpt-4o" 
## "gemini-2.5-pro-exp-03-25"
stage="stage3"
# just for save path
date="0415"
# just for save path
dataset="DS-MVTec"
# just for save path

sample_num=4

fine_prompt_flag=2
# 0 for none, 1 for cot prompt, 2 for detail prompt
demo_text_flag=2
# 0 for none, 1 for caption, 2 for caption and visual_intro
random_demo_flag=0
# 0 for none, 2 for exchange demo's text, 2 for fabrictae demo's text, 3 for swap image with blank
jsonl_path="processed_data/stage2/dataset/2025-04-15/Manufacture_DS-MVTec_1.1_0_4_20:56:03.918875_each_1_shots_demotextflag2_finepromptflag2_exchangedemoflag3.jsonl"
# reading dataset path
success_path="log/"$stage"/success_exp/success_exp_record_"$dataset"_"$date".jsonl"
# succeed exp save dir

echo "Model name:$name"
echo "Success path:$success_path"
echo "Sample number:$sample_num"

python3 src/main.py --stage $stage --success_path $success_path --benchmark_dir $jsonl_path --vlm $model_name --sample_num $sample_num --shot_num 1 --level "detail" --refine_prompt_flag $fine_prompt_flag --demo_text_flag $demo_text_flag --random_demo_flag $random_demo_flag