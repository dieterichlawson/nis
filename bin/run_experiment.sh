#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

DATASET=static_mnist
PROPOSAL=bernoulli_vae
MODEL=nis
LATENT_DIM=50
LOGDIR=/tmp/experiments
GPU=0
REPARAM_PROPOSAL="false"
NAME_SUFFIX=""

print_usage(){
  printf "run_experiment -d <dataset> -p <proposal> -m <model> -l <logdir>  -g <gpu> -r -n <name
  suffix>\n"
}

while getopts 'd:p:m:l:g:n:rh' flag; do
  case "${flag}" in 
    d) DATASET="${OPTARG}" ;;
    p) PROPOSAL="${OPTARG}" ;;
    m) MODEL="${OPTARG}" ;;
    n) NAME_SUFFIX="${OPTARG}" ;;
    l) LOGDIR="${OPTARG}" ;;
    g) GPU="${OPTARG}" ;;
    r) REPARAM_PROPOSAL="true" ;;
    *) print_usage
      exit 1 ;;
  esac
done

if [ "${PROPOSAL}" = "bernoulli_vae" ]; then
  if [ "${REPARAM_PROPOSAL}" = "true" ]; then
    NAME=${PROPOSAL}_proposal_${MODEL}_model_with_prior_grads
  else
    NAME=${PROPOSAL}_proposal_${MODEL}_model_no_prior_grads
  fi
else
  NAME=${PROPOSAL}_proposal_${MODEL}_model
fi

NAME=${NAME}${NAME_SUFFIX}
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
  --reparameterize_proposal=$REPARAM_PROPOSAL \
  --batch_size=128 \
  --max_steps=10000000 >> ${TEXT_OUTDIR}_train.out 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU python3 mnist.py \
  --logdir=$LOGDIR \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=$MODEL \
  --mode=eval \
  --latent_dim=$LATENT_DIM \
  --batch_size=32 \
  --max_steps=10000000 \
  --split=train,valid,test \
  --num_iwae_samples=1,1,1000 >> ${TEXT_OUTDIR}_eval.out 2>&1 &

wait


