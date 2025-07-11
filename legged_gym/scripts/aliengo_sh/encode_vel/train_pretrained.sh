#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
PRETRAIN=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python legged_gym/scripts/train.py --task=aliengo_amp --headless --seed=${SEED} \
--output_name="${CACHE}" \
--checkpoint_model="${PRETRAIN}"/model_16400.pt \
--runner_class_name=EncodeVelAMPOnPolicyRunner \
${EXTRA_ARGS}