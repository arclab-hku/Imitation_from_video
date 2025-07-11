#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python legged_gym/scripts/train.py --task=aliengo_amp --headless --seed=${SEED} \
--output_name="${CACHE}" \
--runner_class_name=ProprioBaseAMPOnPolicyRunner \
${EXTRA_ARGS}