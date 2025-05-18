
industry="Manufacture"
dataset="VisA"
benchmark_dir="/mnt/cfs_bj/liannan/visualicl_raw_data/$industry/"$dataset"/"
candle  capsules  cashew  chewinggum  fryum  macaroni1  macaroni2  pcb1  pcb2  pcb3  pcb4  pipe_fryum
object_list=("candle" "capsule" "cashew" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum")
benchmark_postfix="/test"
sparse_level_list=(1 2 3)

for sparse_level in "${sparse_level_list[@]}"
do
    for object in "${object_list[@]}"
    do  
        echo "***Benchmark Dir:"$benchmark_dir$object$benchmark_postfix
        python3 src/img_process.py --benchmark_dir $benchmark_dir$object$benchmark_postfix --sparse_level $sparse_level
    done
done