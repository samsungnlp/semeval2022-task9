#!/bin/bash

ROOT=$(dirname "${0}")/../..
ROOT=$(realpath "${ROOT}")
echo ROOT dir: "${ROOT}"
export PYTHONPATH=$ROOT

# shellcheck source="${ROOT}"/venv/bin/activate
. "${ROOT}"/venv/bin/activate

export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib:"${LD_LIBRARY_PATH}"

SOURCES_PATH=$ROOT/src

INCLUDE_INGREDIENTS="false"

for i in "$@"; do
  case $i in
    --model=*) MODEL="${i#*=}"; shift;; # default: ahotrod/electra_large_discriminator_squad2_512
    --include_ingredients) INCLUDE_INGREDIENTS="true"; shift;; # whether to include ingredients in the training examples
    --gpu=*) GPU="${i#*=}"; shift;; # GPU (e.g. 0,1), default: ""
  esac
done

if [[ $GPU ]]; then
  echo GPUs: "${GPU}"
  # shellcheck source=$ROOT/scripts/development/gpu_env_setup.sh
  export CUDA_VISIBLE_DEVICES=$GPU
  export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib:"${LD_LIBRARY_PATH}"
else
  echo Running on CPU
  export CUDA_VISIBLE_DEVICES=""
fi

if [[ ! $MODEL ]]; then MODEL="ahotrod/electra_large_discriminator_squad2_512"; fi
if [[ $INCLUDE_INGREDIENTS == "true" ]]; then INCLUDE_INGREDIENTS="--include_ingredients";
                                         else INCLUDE_INGREDIENTS=""; fi

python -W ignore \
       "${SOURCES_PATH}"/reading_comprehension/extractive_qa_engine.py \
       --config_path src/reading_comprehension/configs/train_config.yml \
       --model_name_or_path "${MODEL}" \
       "${INCLUDE_INGREDIENTS}"
