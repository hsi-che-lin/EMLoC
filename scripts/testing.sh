#!/bin/bash

output_name=`date +%Y-%m-%d/%H-%M-%S`
output_dir="output/testing/${output_name}"
ckpt="output/finetune/demo"
data="data/WebSRC/test.json"
plugback="pretrained/InternVL2_5-8B"
zeroshot="False"
generation="config/generation/default.json"

output_dir=${1:-${output_dir}}
ckpt=${2:-${ckpt}}
data=${3:-${data}}
plugback=${4:-${plugback}}
zeroshot=${5:-${zeroshot}}
generation=${6:-${generation}}

mkdir -p "${output_dir}/log-std"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python internvl/engine/testing.py \
  --output_dir "${output_dir}" \
  --ckpt "${ckpt}" \
  --meta_path "${data}" \
  --plugback "${plugback}" \
  --zeroshot "${zeroshot}" \
  --generation "${generation}" \
  --training_args "config/training/default.json" \
  > >(tee -a "${output_dir}/log-std/stdout.log") 2> >(tee -a "${output_dir}/log-std/stderr.log" >&2)
