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

END2END="false"
INCLUDE_INGREDIENTS="false"

for i in "$@"; do
  case $i in
    --end2end) END2END="true"; shift;; # whether to run end-to-end prediction pipeline
    --which=*) WHICH="${i#*=}"; shift;; # which set to evaluate
    --model=*) MODEL="${i#*=}"; shift;; # model_path, default: ahotrod/electra_large_discriminator_squad2_512
    --epoch=*) EPOCH="${i#*=}"; shift;; # model epoch, default: 5
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

if [[ $END2END == "true" ]]; then END2END="--run_end_to_end_prediction"; else END2END=""; fi

if [[ $WHICH == "train" ]]; then
  CONFIG="src/reading_comprehension/configs/predict_train_set_config.yml"
elif [[ $WHICH == "val" ]]; then
  CONFIG="src/reading_comprehension/configs/predict_val_set_config.yml"
elif [[ $WHICH == "test" ]]; then
  CONFIG="src/reading_comprehension/configs/predict_test_set_config.yml"
else
  exit 2
fi

if [[ ! $MODEL ]]; then MODEL="ahotrod/electra_large_discriminator_squad2_512"; fi
if [[ ! $EPOCH ]]; then EPOCH=5; fi
if [[ $INCLUDE_INGREDIENTS == "true" ]]; then INCLUDE_INGREDIENTS="--include_ingredients";
                                         else INCLUDE_INGREDIENTS=""; fi

python -W ignore \
       "${SOURCES_PATH}"/reading_comprehension/extractive_qa_engine.py \
       --config_path "${CONFIG}" \
       --model_name_or_path "${ROOT}/${MODEL}/checkpoint-${EPOCH}" \
       "${END2END}" \
       "${INCLUDE_INGREDIENTS}"

if [[ $END2END ]]
then
  python "${ROOT}"/bin/run_end_to_end_prediction.py  --which "${WHICH}"
fi
