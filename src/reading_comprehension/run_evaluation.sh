#!/bin/bash
echo 1
exit 2
echo 2
ROOT=$(dirname "${0}")/../..
ROOT=$(realpath "${ROOT}")
echo ROOT dir: "${ROOT}"
export PYTHONPATH=$ROOT

# shellcheck source="${ROOT}"/venv/bin/activate
. "${ROOT}"/venv/bin/activate

export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib:"${LD_LIBRARY_PATH}"

SOURCES_PATH=$ROOT/src

for i in "$@"; do
  case $i in
    --which==*) WHICH="${i#*=}"; shift;; # which set to evaluate
    --model=*) MODEL="${i#*=}"; shift;; # default: ahotrod/electra_large_discriminator_squad2_512
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

if [[ $WHICH == "train" ]]; then
  CONFIG="src/reading_comprehension/configs/eval_train_set_config.yml"
elif [[ $WHICH == "val" ]]; then
  CONFIG="src/reading_comprehension/configs/eval_val_set_config.yml"
elif [[ $WHICH == "test" ]]; then
  CONFIG="src/reading_comprehension/configs/eval_test_set_config.yml"
else
  exit 2
fi

if [[ ! $MODEL ]]; then MODEL="ahotrod/electra_large_discriminator_squad2_512"; fi

python -W ignore \
       "${SOURCES_PATH}"/extractive_question_answering.py \
       --config_path "${CONFIG}" \
       --model_name_or_path /home/k.firlag/projects/semeval_qa/results/ahotrod_electra_large_discriminator_squad2_512_include_ingredients/checkpoint-${epoch}
       --model_name_or_path "${MODEL}"
