#!/bin/bash

dataset="manufacture-final"
log_ws="/mnt/cfs_bj/liannan/visualicl_logs/"
# just for save path
industry="Manufacture" 
# just for save path
threshold=0.21       
# filter the high score data (easy ones)
least_model_num=4   
# at least 2 models eval the data
sample_num=1050
shots_num=5         
# 1 for each
shot_target="each"  
# only each status now
demo_text_flag=2    
# (the bigger cover the smaller) 0 for none, 1 for caption, 2 for caption and visual_intro
fine_prompt_flag=2 
# (the bigger cover the smaller) 0 for none, 1 for cot prompt, 2 for detail prompt
random_demo_flag=4
# (the bigger cover the smaller) 0 for none, 1 for exchange demo's text, 2 for fabrictae demo's text, 3 for swap image with blank, 4 for swap image with noise
mask_demo_flag=0
# 0 for none, 1 for mask demo with noise
success_dir=$log_ws"log/stage1/success_exp/"$industry"/"
# combine all exp results in this dir
dataset_path=$"/mnt/cfs_bj/liannan/visualicl_logs/processed_data/stage2/dataset/2025-04-24/Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4_maskdemoflag1.jsonl"
#dataset_path="1"   
# none for create a new dataset from success_dir, actual file for skip the diffculty filter or shots filter
python3 src/filter_pipeline.py --log_ws $log_ws --dataset_path $dataset_path --dataset $dataset --threshold $threshold --sample_num $sample_num --least_model_num $least_model_num --industry $industry --success_dir $success_dir --shots_num $shots_num --shot_target $shot_target --demo_text_flag $demo_text_flag --fine_prompt_flag $fine_prompt_flag --random_demo_flag $random_demo_flag --mask_demo_flag $mask_demo_flag