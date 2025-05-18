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
dataset="BTAD"
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
object_list=("01" "02" "03")
benchmark_postfix="/test"
level="detail"
# where you store your images, please keep it in the following format
## raw_data/<industry>/<dataset>/<object>/<category>/<image>
## Please change object_list manully at least know what objects you have, 
## Don't forget the postfix!

success_path=$log_ws"log/"$stage"/success_exp/$industry/success_exp_record_"$dataset"_"$date".jsonl"
# where the successful exp will be saved, stage2 will take in these jsonl in the same directory

echo "Model name:$model_name"
echo "Success path:$success_path"
echo "Sample number:$sample_num"

for object in "${object_list[@]}"
do  
    echo "***<"$i"> Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
    python3 src/main.py --stage $stage --log_ws $log_ws --success_path $success_path --benchmark_dir $benchmark_dir$object$benchmark_postfix --vlm $model_name --sample_num $sample_num --shot_num 0 --level $level
    i=$((i+1))
done

industry="Manufacture"
dataset="MVTec-LOCO"
object_list=("breakfast_box" "juice_bottle" "pushpins" "screw_bag" "splicing_connectors")
benchmark_postfix="/test"
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

industry="Manufacture"
dataset="ISP-AD"
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
object_list=("ASM")
#object_list=("type10cam2" "type3cam1" "type6cam2" "type9cam2" "type1cam1" "type4cam2" "type7cam2" "type2cam2" "type5cam2" "type8cam1")
benchmark_postfix="/test"
level="detail"
# where you store your images, please keep it in the following format
## raw_data/<industry>/<dataset>/<object>/<category>/<image>
## Please change object_list manully at least know what objects you have, 
## Don't forget the postfix!

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

industry="Manufacture"
dataset="MVTec-3D-new"
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
object_list=("cable_gland" "dowel" "foam" "rope" "tire")
#object_list=("type10cam2" "type3cam1" "type6cam2" "type9cam2" "type1cam1" "type4cam2" "type7cam2" "type2cam2" "type5cam2" "type8cam1")
benchmark_postfix="/test_rgb"
level="detail"
# where you store your images, please keep it in the following format
## raw_data/<industry>/<dataset>/<object>/<category>/<image>
## Please change object_list manully at least know what objects you have, 
## Don't forget the postfix!

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


industry="Manufacture"
dataset="ITD"
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
object_list=("type10cam2" "type3cam1" "type6cam2" "type9cam2" "type1cam1" "type4cam2" "type7cam2" "type2cam2" "type5cam2" "type8cam1")
benchmark_postfix="/test"
level="detail"
# where you store your images, please keep it in the following format
## raw_data/<industry>/<dataset>/<object>/<category>/<image>
## Please change object_list manully at least know what objects you have, 
## Don't forget the postfix!

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

industry="Manufacture"
dataset="GoodsAD"
object_list=("cigarette_box" "drink_bottle" "drink_can" "food_bottle" "food_box" "food_package")
benchmark_postfix="/test"
level="element"

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

industry="Manufacture"
dataset="MVTec-AD"
object_list=("bottle" "cable" "capsule" "grid" "metal_nut" "pill" "screw" "tile" "transistor" "wood")
benchmark_postfix="/test"
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

industry="Manufacture"
dataset="NEU-DET"
object_list=("steelstrips")
level="pattern"

benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
success_path=$log_ws"log/"$stage"/success_exp/$industry/success_exp_record_"$dataset"_"$date".jsonl"
# where the successful exp will be saved, stage2 will take in these jsonl in the same directory

echo "Model name:$model_name"
echo "Success path:$success_path"
echo "Sample number:$sample_num"

for object in "${object_list[@]}"
do  
    echo "***"$i"Benchmark Dir:"$benchmark_dir$object
    python3 src/main.py --stage $stage --log_ws $log_ws --success_path $success_path --benchmark_dir $benchmark_dir$object --vlm $model_name --sample_num $sample_num --shot_num 0 --level $level
    i=$((i+1))
done

industry="Manufacture"
dataset="VisA"
object_list=("candle" "capsules" "cashew" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum")
benchmark_postfix="/test"
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
