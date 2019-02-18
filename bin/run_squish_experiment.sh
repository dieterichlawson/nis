#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

DATASET=raw_mnist
PROPOSAL=gaussian_vae
MODEL=nis
LATENT_DIM=50
LOGDIR=/tmp/experiments/squash
GPU=0
NAME_SUFFIX=""
SQUASH="false"

print_usage(){
  printf "run_experiment -l <logdir>  -g <gpu> -s -n <name suffix>\n"
}

while getopts 'l:g:n:sh' flag; do
  case "${flag}" in 
    n) NAME_SUFFIX="${OPTARG}" ;;
    l) LOGDIR="${OPTARG}" ;;
    g) GPU="${OPTARG}" ;;
    s) SQUASH="true" ;;
    *) print_usage
      exit 1 ;;
  esac
done

if [ "${SQUASH}" = "true" ]; then
  NAME=${PROPOSAL}_proposal_${MODEL}_model_squashed${NAME_SUFFIX}
else
  NAME=${PROPOSAL}_proposal_${MODEL}_model_not_squashed${NAME_SUFFIX}
fi

LOGDIR=$LOGDIR/$DATASET/$NAME
TEXT_OUTDIR=logdir/$NAME

CUDA_VISIBLE_DEVICES=$GPU python3 mnist.py \
  --logdir=$LOGDIR  \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=$MODEL \
  --mode=train \
  --learning_rate=3e-4 \
  --decay_lr \
  --anneal_kl_step=100000 \
  --latent_dim=$LATENT_DIM \
  --reparameterize_proposal=true \
  --squash=$SQUASH \
  --batch_size=128 \
  --max_steps=10000000 >> ${TEXT_OUTDIR}_train.out 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU python3 mnist.py \
  --logdir=$LOGDIR \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=$MODEL \
  --mode=eval \
  --latent_dim=$LATENT_DIM \
  --squash=$SQUASH \
  --batch_size=32 \
  --max_steps=10000000 \
  --split=train,valid,test \
  --num_iwae_samples=1,1,1000 >> ${TEXT_OUTDIR}_eval.out 2>&1 &

wait


