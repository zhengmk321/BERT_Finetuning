#!/bin/bash

# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-1"
# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-2"
# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-3"
# ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-4"
ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-6"
out="../results/bert_pretraining/SQuAD"
names=$( hostname )

arr=($(echo "$names" | tr '.' '\n'))
server="${arr[1]}"


# for first_dir in $ckpt_dir; do
    for sec_dir in $ckpt_dir; do
        files=($sec_dir/*.pt)
        for ((i=${#files[@]}-1; i>=0; i--)); do
            FILE="${files[$i]}"

            arr=($(echo "$FILE" | tr '/' '\n'))
            ckpt_file="${arr[-1]}"

            folder_name="${arr[-2]}"

            arr=($(echo "$ckpt_file" | tr '.' '\n'))
            ckpt_name="${arr[0]}"

            out_dir="$out/$folder_name/$ckpt_name" 
            
            CMD="./run_squad.sh $FILE $out_dir"
            $CMD
            # find $out_dir -name \*.bin -type f -delete

        done
    done
# done

