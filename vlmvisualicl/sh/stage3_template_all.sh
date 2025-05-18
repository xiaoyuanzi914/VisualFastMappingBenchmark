#!/bin/bash
log_ws="/mnt/cfs_bj/liannan/visualicl_logs/"
# model chocie 
model_list=("random-policy" "doubao-1.5-vision-pro-32k-250115" "Qwen*Qwen2.5-VL-72B-Instruct" "Qwen*QVQ-72B-Preview" "deepseek-ai*deepseek-vl2" "gemini-2.5-pro-exp-03-25" "gpt-4o" "gpt-4.1" "<gpt>o3" "<gpt>o4-mini" "claude-3-7-sonnet-20250219" "claude-3-7-sonnet-20250219-thinking" "ernie-4.5-8k-preview" "ernie-4.5-turbo-vl-32k" "internvl-78b")
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
## 13 for "ernie-4.5-turbo-vl-32k"
## 14 for "internvl-78b"
model_name=${model_list[$1]}
echo $model_name
stage="stage3"
# just for save path
date="0424"
# just for save path
dataset="manufacture-final"
# just for save path

sample_num=1050

jsonl_path="/mnt/cfs_bj/liannan/visualicl_logs/processed_data/stage2/dataset/2025-04-24/Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4_maskdemoflag1_each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4.jsonl"
# # reading dataset path
success_path=$log_ws"log/"$stage"/success_exp/success_exp_record_"$dataset"_"$date".jsonl"
# # succeed exp save dir

shot_num=1
fine_prompt_flag_list=(0)
# 0 for none, 1 for cot prompt, 2 for detail prompt
demo_text_flag_list=(0)
# 0 for none, 1 for caption, 2 for visual_intro
random_demo_flag_list=(0 1 2 3 4 5 6)
# 0 for none, 1 for exchange demo's text, 2 for fabrictae demo's text and candidate, 3 for swap image with blank, 4 for swap image with noise, 5 for add noise with nega demo, 6 for replicate among demo
mask_demo_flag_list=(-1 0 1 2 3)
# -1 for enhence, 0 for none, 1 for mask query with 100% noise, 2 for mask query with 50% noise, 3 for mask query with 33% noise

# max exp: acc vs shots
shot_num_list=(0 1 2 3 4 5)
fine_prompt_flag=0
demo_text_flag=0
random_demo_flag=0
mask_demo_flag=0
for shot_num in "${shot_num_list[@]}"
do
    echo "---***---"$shot_num"-"$fine_prompt_flag"-"$demo_text_flag"-"$random_demo_flag"-"$mask_demo_flag"---***---"
    python3 src/main.py --log_ws $log_ws --stage $stage --success_path $success_path --benchmark_dir $jsonl_path --vlm $model_name --sample_num $sample_num --shot_num $shot_num --level "detail" --refine_prompt_flag $fine_prompt_flag --demo_text_flag $demo_text_flag --random_demo_flag $random_demo_flag --mask_demo_flag $mask_demo_flag
done

# mechanism exp
# - acc analysis for prior llm knowledge(2)
# - acc analysis for Intra-Demo Homogeneity(6)
# - acc analysis for Inter-class Distinctiveness(5)
shot_num_list=(0 1 2 3 4 5)
fine_prompt_flag=0
demo_text_flag=0
random_demo_flag_list=(2 5 6)
mask_demo_flag=0
for random_demo_flag in "${random_demo_flag_list[@]}"
do
    for shot_num in "${shot_num_list[@]}"
    do
        echo "---***---"$shot_num"-"$fine_prompt_flag"-"$demo_text_flag"-"$random_demo_flag"-"$mask_demo_flag"---***---"
        python3 src/main.py --log_ws $log_ws --stage $stage --success_path $success_path --benchmark_dir $jsonl_path --vlm $model_name --sample_num $sample_num --shot_num $shot_num --level "detail" --refine_prompt_flag $fine_prompt_flag --demo_text_flag $demo_text_flag --random_demo_flag $random_demo_flag --mask_demo_flag $mask_demo_flag
    done
done

# cause location exp
# - acc analysis for fine grain detection
mask_demo_flag_list=(-1 0)
shot_num_list=(0 1 2 3 4 5)
fine_prompt_flag=0
demo_text_flag=0
random_demo_flag=0
for shot_num in "${shot_num_list[@]}"
do
    for mask_demo_flag in "${mask_demo_flag_list[@]}"
    do
        echo "---***---"$shot_num"-"$fine_prompt_flag"-"$demo_text_flag"-"$random_demo_flag"-"$mask_demo_flag"---***---"
        python3 src/main.py --log_ws $log_ws --stage $stage --success_path $success_path --benchmark_dir $jsonl_path --vlm $model_name --sample_num $sample_num --shot_num $shot_num --level "detail" --refine_prompt_flag $fine_prompt_flag --demo_text_flag $demo_text_flag --random_demo_flag $random_demo_flag --mask_demo_flag $mask_demo_flag
    done
done

# - acc analysis for lack of knowledge 1

shot_num=0
fine_prompt_flag_list=(0 2)
demo_text_flag=0
random_demo_flag=0
mask_demo_flag=0

for fine_prompt_flag in "${fine_prompt_flag_list[@]}"
do
    echo "---***---"$shot_num"-"$fine_prompt_flag"-"$demo_text_flag"-"$random_demo_flag"-"$mask_demo_flag"---***---"
    python3 src/main.py --log_ws $log_ws --stage $stage --success_path $success_path --benchmark_dir $jsonl_path --vlm $model_name --sample_num $sample_num --shot_num $shot_num --level "detail" --refine_prompt_flag $fine_prompt_flag --demo_text_flag $demo_text_flag --random_demo_flag $random_demo_flag --mask_demo_flag $mask_demo_flag
done

# - acc analysis for lack of knowledge 2
shot_num=0
fine_prompt_flag=0
demo_text_flag_list=(0 1)
random_demo_flag=0
mask_demo_flag=0

for demo_text_flag in "${demo_text_flag_list[@]}"
do
    echo "---***---"$shot_num"-"$fine_prompt_flag"-"$demo_text_flag"-"$random_demo_flag"-"$mask_demo_flag"---***---"
    python3 src/main.py --log_ws $log_ws --stage $stage --success_path $success_path --benchmark_dir $jsonl_path --vlm $model_name --sample_num $sample_num --shot_num $shot_num --level "detail" --refine_prompt_flag $fine_prompt_flag --demo_text_flag $demo_text_flag --random_demo_flag $random_demo_flag --mask_demo_flag $mask_demo_flag
done
