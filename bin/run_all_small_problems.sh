#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

LOGDIR=/home/dieterich_lawson/small_experiments/
ENERGY_FN_SIZES=20,20
LR=0.0005
TEXT_OUTDIR=logdir

# CHECKERBOARD

python3 small_problems.py \
  --target=checkerboard  \
  --algo=nis \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/checkerboard/nis >> ${TEXT_OUTDIR}/checkerboard_nis.log 2>&1 &

python3 small_problems.py \
  --target=checkerboard  \
  --algo=lars \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/checkerboard/lars >> ${TEXT_OUTDIR}/checkerboard_lars.log 2>&1 &

python3 small_problems.py \
  --target=checkerboard  \
  --algo=his \
  --his_T=5 \
  --his_learn_stepsize \
  --his_learn_alpha \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/checkerboard/his >> ${TEXT_OUTDIR}/checkerboard_his.log 2>&1 &


# TWO RINGS

python3 small_problems.py \
  --target=two_rings \
  --algo=nis \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/two_rings/nis >> ${TEXT_OUTDIR}/two_rings_nis.log 2>&1 &

python3 small_problems.py \
  --target=two_rings \
  --algo=lars \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/two_rings/lars >> ${TEXT_OUTDIR}/two_rings_lars.log 2>&1 &

python3 small_problems.py \
  --target=two_rings \
  --algo=his \
  --his_T=5 \
  --his_learn_stepsize \
  --his_learn_alpha \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/two_rings/his >> ${TEXT_OUTDIR}/two_rings_his.log 2>&1 &


# NINE GAUSSIANS

python3 small_problems.py \
  --target=nine_gaussians \
  --nine_gaussians_variance=0.01 \
  --algo=nis \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/nine_gaussians/nis >> ${TEXT_OUTDIR}/nine_gaussians_nis.log 2>&1 &

python3 small_problems.py \
  --target=nine_gaussians \
  --nine_gaussians_variance=0.01 \
  --algo=lars \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/nine_gaussians/lars >> ${TEXT_OUTDIR}/nine_gaussians_lars.log 2>&1 &

python3 small_problems.py \
  --target=nine_gaussians \
  --nine_gaussians_variance=0.01 \
  --algo=his \
  --his_T=5 \
  --his_learn_stepsize \
  --his_learn_alpha \
  --batch_size=128 \
  --learning_rate=${LR} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/nine_gaussians/his >> ${TEXT_OUTDIR}/nine_gaussians_his.log 2>&1 &

wait
