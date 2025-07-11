#!/bin/bash
CACHE=$1
python legged_gym/scripts/play.py --task=aliengo_amp \
--load_run="${CACHE}" --checkpoint=-1 \
--export_policy --export_onnx_policy


