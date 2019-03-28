#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

LOGDIR=/tmp/small_experiments/
ENERGY_FN_SIZES=100,100
DENSITY_NUM_POINTS=250
LR=0.0005
TEXT_OUTDIR=logdir

# CHECKERBOARD

python3 small_problems.py \ 
  --target=checkerboard  \
  --algo=nis \
  --batch_size=128 \
  --learning_rate=${LR} \
  --density_num_points=${DENSITY_NUUM_POINTS} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/checkerboard/nis >> ${TEXT_OUTDIR}/checkerboard_nis.log 2>&1 &

python3 small_problems.py \ 
  --target=checkerboard  \
  --algo=lars \
  --batch_size=128 \
  --learning_rate=${LR} \
  --density_num_points=${DENSITY_NUUM_POINTS} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/checkerboard/lars >> ${TEXT_OUTDIR}/checkerboard_lars.log 2>&1 &

# TWO RINGS

python3 small_problems.py \ 
  --target=two_rings \
  --algo=nis \
  --batch_size=128 \
  --learning_rate=${LR} \
  --density_num_points=${DENSITY_NUUM_POINTS} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/two_rings/nis >> ${TEXT_OUTDIR}/two_rings_nis.log 2>&1 &

python3 small_problems.py \ 
  --target=two_rings \
  --algo=lars \
  --batch_size=128 \
  --learning_rate=${LR} \
  --density_num_points=${DENSITY_NUUM_POINTS} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/two_rings/lars >> ${TEXT_OUTDIR}/two_rings_lars.log 2>&1 &

# NINE GAUSSIANS

python3 small_problems.py \ 
  --target=nine_gaussians \
  --algo=nis \
  --batch_size=128 \
  --learning_rate=${LR} \
  --density_num_points=${DENSITY_NUUM_POINTS} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/nine_gaussians/nis >> ${TEXT_OUTDIR}/nine_gaussians_nis.log 2>&1 &

python3 small_problems.py \ 
  --target=nine_gaussians \
  --algo=lars \
  --batch_size=128 \
  --learning_rate=${LR} \
  --density_num_points=${DENSITY_NUUM_POINTS} \
  --energy_fn_sizes=${ENERGY_FN_SIZES} \
  --max_steps=10000000 \
  --logdir=${LOGDIR}/nine_gaussians/lars >> ${TEXT_OUTDIR}/nine_gaussians_lars.log 2>&1 &

wait
