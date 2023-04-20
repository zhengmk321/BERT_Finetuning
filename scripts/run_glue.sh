#!/bin/bash

MODEL_CHECKPOINT=${1:-"/work/09308/zhengmk/BERT_pretrained_model/ckpt_8601.ckpt"}
OUTPUT_DIR=${2:-"../results/bert_pretraining/GLUE"}
CONFIG_FILE=${3:-"../config/bert_large_uncased_config.json"}

DATA_DIR="/work/09308/zhengmk/BERT_data/"
COLA_DIR="$DATA_DIR/CoLA"

TASK_NAME="cola"

BERT_MODEL="bert-large-uncased"

LOGFILE="$OUTPUT_DIR/$TASK_NAME""_log.txt"

NGPUS=8
BATCH_SIZE=16

echo "Output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
if [ ! -d "$OUTPUT_DIR" ]; then
	echo "ERROR: unable to make $OUTPUT_DIR"
fi

CMD="torchrun --nproc_per_node=$NGPUS ../run_glue.py "

CMD+=" --init_checkpoint=$MODEL_CHECKPOINT "
 
CMD+=" --bert_model=$BERT_MODEL "
CMD+=" --data_dir=$COLA_DIR "
CMD+=" --task_name=$TASK_NAME "

CMD+=" --do_train "
CMD+=" --train_batch_size=$BATCH_SIZE "

CMD+=" --do_lower_case "
CMD+=" --bert_model=$BERT_MODEL "
CMD+=" --config_file=$CONFIG_FILE "

CMD+=" --do_predict "
CMD+=" --predict_batch_size=$BATCH_SIZE "
CMD+=" --do_eval "

CMD+=" --max_seq_length=128 "
CMD+=" --learning_rate=2e-5 "
CMD+=" --num_train_epochs=3 "
CMD+=" --output_dir=$OUTPUT_DIR "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --fp16 "

echo "$CMD | tee $LOGFILE"
time $CMD | tee $LOGFILE