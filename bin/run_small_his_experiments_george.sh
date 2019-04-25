#!/bin/bash

PROBLEM=nine_gaussians
BATCH_SIZE=128
LOGDIR=log/experiments
LR=0.0003
TEXT_OUTDIR=logdir

step_size=0.1
T=5

ALGO=his
#ALGO=nis

RUN_NAME=${ALGO}

python3 small_problems.py \
      --algo=$ALGO \
      --target=$PROBLEM \
      --energy_fn_sizes=20,20 \
      --his_t=${T} \
      --his_stepsize=${step_size} \
      --learning_rate=${LR} \
      --batch_size=${BATCH_SIZE} \
      --logdir=${LOGDIR}/${PROBLEM}/${RUN_NAME} #>> ${TEXT_OUTDIR}/${RUN_NAME}.log 2>&1 &

