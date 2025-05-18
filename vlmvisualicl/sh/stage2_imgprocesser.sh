
industry="Manufacture"

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