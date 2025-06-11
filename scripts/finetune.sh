#!/bin/bash

output_name=`date +%Y-%m-%d/%H-%M-%S`
output_dir="output/finetune/${output_name}"
model="pretrained/WebSRC/InternVL2_5-8B/8B-to-2B-64"
data="data/WebSRC/train.json"
peft="config/peft/8B.json"
quantization="config/quantization/default.json"
training="config/training/500-steps.json"
use_svd="True"
use_flash_attn="False"

output_dir=${1:-${output_dir}}
model=${2:-${model}}
data=${3:-${data}}
peft=${4:-${peft}}
quantization=${5:-${quantization}}
training=${6:-${training}}
use_svd=${7:-${use_svd}}
use_flash_attn=${8:-${use_flash_attn}}

mkdir -p "${output_dir}/log-std"

GPUS=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/engine/finetune.py \
  --output_dir "${output_dir}" \
  --model_name_or_path "${model}" \
  --meta_path "${data}" \
  --peft_config "${peft}" \
  --quantization_config "${quantization}" \
  --training_args "${training}" \
  --use_svd "${use_svd}" \
  --use_flash_attn "${use_flash_attn}" \
  > >(tee -a "${output_dir}/log-std/stdout.log") 2> >(tee -a "${output_dir}/log-std/stderr.log" >&2)
