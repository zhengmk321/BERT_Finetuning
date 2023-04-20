#!/bin/bash
# ckpt_dir1="/work2/00946/zzhang/ckpts/eb1e-3"
# ckpt_dir2="/work2/00946/zzhang/ckpts/eb1e-5"
# ckpt_dir="/work/09308/zhengmk/BERT_pretrained_model/lamb-kfac/"
ckpt_dir="/scratch/00946/zzhang/data/bert/lamb-ckpt/"
out="../results/bert_pretraining/GLUE"
names=$( hostname )

arr=($(echo "$names" | tr '.' '\n'))
server="${arr[1]}"


# if [ $server = 'frontera' ]; then
#   num_gpu_per_node='4'
# elif [ $server = 'ls6' ]; then
#   num_gpu_per_node='2'
# else
#   num_gpu_per_node='1'
# fi
num_gpu_per_node='2'
total_num_gpus='2'
# num_nodes=$(($total_num_gpus/$num_gpu_per_node))
num_nodes=1

for ckpt_dir in $ckpt_dir; do
  for FILE in "$ckpt_dir"/*; do

    arr=($(echo "$FILE" | tr '/' '\n'))
    ckpt_file="${arr[-1]}"

    folder_name="${arr[-2]}"

    arr=($(echo "$ckpt_file" | tr '.' '\n'))
    ckpt_name="${arr[0]}"

    for seed in 0 1 2 3 4 5 6 7 8 9; do
      out_dir="$out/$folder_name/$ckpt_name" 
      out_dir+="_seed_$seed"
      for task in 'MRPC' 'SST-2'; do
        if [ $task = 'MRPC' ]; then
          batch_size="64"
        elif [ $task = 'SST-2' ]; then
          batch_size="512"
        else 
          batch_size="16"
        fi
        CMD="./run_glue.sh $task $FILE $num_nodes $num_gpu_per_node $batch_size $out_dir $seed"
        $CMD
      done
    done    
    
  done
done


