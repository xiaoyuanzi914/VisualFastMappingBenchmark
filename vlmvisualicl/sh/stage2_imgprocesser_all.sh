
industry="Manufacture"

dataset="ISP-AD"
object_list=("ASM")
benchmark_postfix="/test"
sparse_level_list=(1 2 3)
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"

for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done


dataset="MVTec-AD"
object_list=("bottle" "cable" "capsule" "grid" "metal_nut" "pill" "screw" "tile" "transistor" "wood")
benchmark_postfix="/test"
sparse_level_list=(1 2 3)
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"

for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done

dataset="MVTec-AD-2"
object_list=("sheet_metal" "vial" "wallplugs")
benchmark_postfix="/test_public"
sparse_level_list=(1 2 3)
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"

for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done


dataset="MVTec-LOCO"
object_list=("breakfast_box" "juice_bottle" "pushpins" "screw_bag" "splicing_connectors")
benchmark_postfix="/test"
sparse_level_list=(1 2 3)
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"

for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done


dataset="VisA"
object_list=("candle" "capsules" "cashew" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum")
benchmark_postfix="/test"
sparse_level_list=(1 2 3)
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"

for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done


dataset="MVTec-3D-new"
object_list=("cable_gland" "dowel" "foam" "rope" "tire")
benchmark_postfix="/test_rgb"
sparse_level_list=(1 2 3)
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"


for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done