#!/bin/bash

python main.py \
  --model_path ../models/llama-2-7b-hf \
  --model_name llama-2-7b-hf \
  --output_dir ./log/llama-2-7b-hf \
  --wbits 16 \
  --input_bits 8 --input_mode static --input_group_size -1 \
  --k_bits 16 --v_bits 16 --kv_mode static --kv_group_size -1 \
  --input_asym \
  --eval_tasks=mmlu_abstract_algebra \
  --eval_ppl \
  --use_insert_quant \
  # --set_prefixed_tokens \
  # --pre_rotate --down_online_had --qk_online_had \