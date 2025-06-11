#!/bin/bash

output_dir="pretrained/WebSRC/InternVL2_5-8B/8B-to-2B-64"
model="pretrained/InternVL2_5-8B"
data="data/WebSRC/train.json"
compress="config/compress/8B-to-4B.json"
scaling="config/calibration/64.json"

output_dir=${1:-${output_dir}}
model=${2:-${model}}
data=${3:-${data}}
compress=${4:-${compress}}
scaling=${5:-${scaling}}

mkdir -p "${output_dir}/log-std"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python internvl/engine/emulatorConstruct.py \
  --output_dir "${output_dir}" \
  --model_name_or_path "${model}" \
  --meta_path "${data}" \
  --compress_config "${compress}" \
  --scaling_config "${scaling}" \
  --quantization_config "config/quantization/default.json" \
  --peft_config "config/peft/default.json" \
  --training_args "config/training/default.json" \
  --use_svd "True" \
  --use_flash_attn "False" \
  > >(tee -a "${output_dir}/log-std/stdout.log") 2> >(tee -a "${output_dir}/log-std/stderr.log" >&2)
