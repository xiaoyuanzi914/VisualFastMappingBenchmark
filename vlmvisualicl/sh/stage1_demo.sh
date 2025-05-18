#!/bin/bash
model_name="doubao-1.5-vision-pro-32k-250115"
# model chocie 
## "doubao-1.5-vision-pro-32k-250115" 
## "Qwen*Qwen2.5-VL-72B-Instruct" 
## "deepseek-ai*deepseek-vl2" 
## "gpt-4o" 
## "gemini-2.5-pro-exp-03-25"
stage="stage1"
# just for save path
date="0415"
# just for save path
sample_num=5

industry="Manufacture"
dataset="DS-MVTec"
benchmark_dir="demo_data/raw_data/$industry/"$dataset"/"
object_list=("bottle" "cable" "capsule" "carpet" "grid")
benchmark_postfix="/image"
# where you store your images, please keep it in the following format
## raw_data/<industry>/<dataset>/<object>/<category>/<image>

success_path="log/"$stage"/success_exp/$industry/success_exp_record_"$dataset"_"$date".jsonl"
# where the successful exp will be saved, stage2 will take in these jsonl in the same directory

echo "Model name:$name"
echo "Success path:$success_path"
echo "Sample number:$sample_num"

for object in "${object_list[@]}"
do  
    echo "Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
    python3 src/main.py --stage $stage --success_path $success_path --benchmark_dir $benchmark_dir$object$benchmark_postfix --vlm $model_name --sample_num $sample_num --shot_num 0 --level "detail"
done
