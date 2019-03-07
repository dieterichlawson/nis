#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

PROBLEM=nine_gaussians
BATCH_SIZE=128
LOGDIR=/tmp/experiments
LR=0.0003
TEXT_OUTDIR=logdir

for step_size in 0.3 0.1 0.03 0.01; do
  for T in 5 25 50; do
    echo "Starting run with step size=${step_size} and T=${T}"
    python3 small_problems.py \
          --algo=his \
          --target=$PROBLEM \
          --energy_fn_sizes=20,20 \
          --his_t=${T} \
          --his_stepsize=${step_size} \
          --learning_rate=${LR} \
          --batch_size=${BATCH_SIZE} \
          --logdir=${LOGDIR}/${PROBLEM}/${RUN_NAME} >> ${TEXT_OUTDIR}/${RUN_NAME}.log 2>&1 &
  done
done

wait
