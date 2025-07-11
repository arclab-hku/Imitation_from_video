#!/bin/bash
CACHE=$1
python legged_gym/scripts/play.py --task=a1_amp \
--load_run="${CACHE}" --checkpoint=17800



