#!/bin/bash

MODEL_CHECKPOINT=${1:-"/work/09308/zhengmk/BERT_pretrained_model/ckpt_8601.pt"}
OUTPUT_DIR=${2:-"../results/bert_pretraining/squad"}
CONFIG_FILE=${3:-"../config/bert_large_uncased_config.json"}

DATA_DIR="/work/09308/zhengmk/BERT_data/"
SQUAD_DIR="$DATA_DIR/squad/v1.1"

BERT_MODEL="bert-large-uncased"

NGPUS=4
BATCH_SIZE=6

LOGFILE="$OUTPUT_DIR/squad_log.txt"

echo "Output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
if [ ! -d "$OUTPUT_DIR" ]; then
	echo "ERROR: unable to make $OUTPUT_DIR"
fi

CMD="torchrun --nproc_per_node=$NGPUS ../run_squad.py"

CMD+=" --init_checkpoint=$MODEL_CHECKPOINT "

CMD+=" --do_train "
CMD+=" --train_file=$SQUAD_DIR/train-v1.1.json "
CMD+=" --train_batch_size=$BATCH_SIZE "

CMD+=" --do_predict "
CMD+=" --predict_file=$SQUAD_DIR/dev-v1.1.json "
CMD+=" --predict_batch_size=$BATCH_SIZE "
CMD+=" --eval_script=$SQUAD_DIR/evaluate-v1.1.py "
CMD+=" --do_eval "

CMD+=" --do_lower_case "
CMD+=" --bert_model=$BERT_MODEL "
CMD+=" --learning_rate=3e-5 "
CMD+=" --num_train_epochs=2 "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUTPUT_DIR "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --fp16"

echo "$CMD | tee $LOGFILE"
time $CMD | tee $LOGFILE
