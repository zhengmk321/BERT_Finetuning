#!/bin/bash
# ckpt_dir1="/work2/00946/zzhang/ckpts/eb1e-3"
# ckpt_dir2="/work2/00946/zzhang/ckpts/eb1e-5"

# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-1"
# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-2"
# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-3"
# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-4"
ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-6"
out="../results/bert_pretraining/GLUE"
names=$( hostname )

arr=($(echo "$names" | tr '.' '\n'))
server="${arr[1]}"


if [ $server = 'frontera' ]; then
  num_gpu_per_node='4'
elif [ $server = 'ls6' ]; then
  num_gpu_per_node='3'
else
  num_gpu_per_node='1'
fi
total_num_gpus=4

# for first_dir in $ckpt_dir; do
    for sec_dir in $ckpt_dir; do
        files=($sec_dir/*.pt)
        for ((i=${#files[@]}-1; i>=0; i--)); do
        # for ((i=0; i<${#files[@]}; i++)); do
            FILE="${files[$i]}"

            arr=($(echo "$FILE" | tr '/' '\n'))
            ckpt_file="${arr[-1]}"

            folder_name="${arr[-2]}"

            arr=($(echo "$ckpt_file" | tr '.' '\n'))
            ckpt_name="${arr[0]}"

            out_dir="$out/$folder_name/$ckpt_name" 
            
            for task in 'MNLI' 'QQP' 'QNLI' 'SST-2'  'CoLA' 'STS-B' 'MRPC' 'RTE'; do
                # if [ $task = 'MRPC' ]; then
                #     batch_size="4"
                # elif [ $task = 'SST-2' ]; then
                #     batch_size="32"
                # else 
                #     batch_size="16"
                # fi
                batch_size=16
                CMD="./run_one_glue_nv.sh $task $FILE  $num_gpu_per_node $batch_size $out_dir"
                $CMD
            done
        done
    done
# done

