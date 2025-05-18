#!/bin/bash
log_ws="/mnt/cfs_bj/liannan/visualicl_logs/"
# model chocie 
model_list=("random-policy" "doubao-1.5-vision-pro-32k-250115" "Qwen*Qwen2.5-VL-72B-Instruct" "Qwen*QVQ-72B-Preview" "deepseek-ai*deepseek-vl2" "gemini-2.5-pro-exp-03-25" "gpt-4o" "gpt-4.1" "<gpt>o3" "<gpt>o4-mini" "claude-3-7-sonnet-20250219" "claude-3-7-sonnet-20250219-thinking" "ernie-4.5-8k-preview" "internvl-78b")
## 0 for "random-policy" 
## 1 for "doubao-1.5-vision-pro-32k-250115" 
## 2 for "Qwen*Qwen2.5-VL-72B-Instruct" 
## 3 for "Qwen*QVQ-72B-Preview"
## 4 for "deepseek-ai*deepseek-vl2"
## 5 for "gemini-2.5-pro-exp-03-25" 
## 6 for "gpt-4o" 
## 7 for "gpt-4.1"
## 8 for "<gpt>o3"
## 9 for "<gpt>o4-mini"
## 10 for "claude-3-7-sonnet-20250219"
## 11 for "claude-3-7-sonnet-20250219-thinking"
## 12 for "ernie-4.5-8k-preview"
## 13 for "internvl-78b"
model_name=${model_list[$1]}
echo $model_name
stage="stage1"
# just for save path
date="0422"
# just for save path
sample_num=1000
i=0

industry="Manufacture"
dataset="MVTec-AD-2"
object_list=("sheet_metal" "vial" "wallplugs")
benchmark_postfix="/test_public"
level="detail"

benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
success_path=$log_ws"log/"$stage"/success_exp/$industry/success_exp_record_"$dataset"_"$date".jsonl"
# where the successful exp will be saved, stage2 will take in these jsonl in the same directory

echo "Model name:$model_name"
echo "Success path:$success_path"
echo "Sample number:$sample_num"

for object in "${object_list[@]}"
do  
    echo "***"$i"Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
    python3 src/main.py --stage $stage --log_ws $log_ws --success_path $success_path --benchmark_dir $benchmark_dir$object$benchmark_postfix --vlm $model_name --sample_num $sample_num --shot_num 0 --level $level
    i=$((i+1))
done