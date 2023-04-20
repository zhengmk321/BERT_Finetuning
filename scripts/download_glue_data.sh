#!/bin/bash

# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Download
# download_wikipedia --outdir ${BERT_PREP_WORKING_DIR}/wikipedia/
# python3 ../utilis/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab
# python3 ../utilis/bertPrep.py --action download --dataset squad
python3 ../utils/bertPrep.py --action download --dataset mrpc
python3 ../utils/bertPrep.py --action download --dataset sst-2
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks CoLA
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks MNLI
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks QQP
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks STS
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks QNLI
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks RTE
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks WNLI
python3 ../utils/download_glue_data.py --data_dir /work/09308/zhengmk/BERT_data/download/glue --tasks MRPC --path_to_mrpc /work/09308/zhengmk/BERT_data/download/glue/MRPC/