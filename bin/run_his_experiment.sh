#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

DATASET=static_mnist
PROPOSAL=gaussian_vae
MODEL=his
LATENT_DIM=50
LOGDIR=/tmp/experiments
GPU=0
NAME_SUFFIX=""
LEARN_TEMP="false"
LEARN_STEPSIZE="false"
INIT_TEMP=1.0
INIT_STEPSIZE=0.01
SQUASH="false"
TIMESTEPS=100

for flag in "$@"; do
case $flag in
    -d=*|--dataset=*)
    DATASET="${flag#*=}"
    shift ;;# past argument=value
    -p=*|--proposal=*)
    PROPOSAL="${flag#*=}"
    shift ;; # past argument=value
    -l=*|--logdir=*)
    LOGDIR="${flag#*=}"
    shift ;;# past argument=value
    -g=*|--gpu=*)
    GPU="${flag#*=}"
    shift ;;
    -n=*|--name_suffix=*)
    NAME_SUFFIX="${flag#*=}"
    shift ;;
    --learn_temp)
    LEARN_TEMP="true"
    shift ;;
    --learn_stepsize)
    LEARN_STEPSIZE="true"
    shift ;;
    --init_temp=*)
    INIT_TEMP="${flag#*=}"
    shift ;;
    --init_stepsize=*)
    INIT_STEPSIZE="${flag#*=}"
    shift ;;
    --squash)
    SQUASH="true"
    shift ;;
    -t=*|--timesteps=*)
    TIMESTEPS="${flag#*=}"
    shift ;;

    *) print_usage ;;
esac
done
echo "Args"
echo "  dataset:" $DATASET
echo "  proposal:" $PROPOSAL
echo "  logdir:" $LOGDIR
echo "  gpu:" $GPU
echo "  name suffix:" $NAME_SUFFIX
echo "  learn temp:" $LEARN_TEMP
echo "  learn stepsize:" $LEARN_STEPSIZE
echo "  init temp:" $INIT_TEMP
echo "  init stepsize:" $INIT_STEPSIZE
echo "  timesteps:" $TIMESTEPS

if [ "${LEARN_TEMP}" = "true" ]; then
  lt_suff="lt_y"
else
  lt_suff="lt_n"
fi

if [ "${LEARN_STEPSIZE}" = "true" ]; then
  ss_suff="lss_y"
else
  ss_suff="lss_n"
fi

if [ "${SQUASH}" = "true" ]; then
  squash_suff="sq_y"
else
  squash_suff="sq_n"
fi

it_suff="it_${INIT_TEMP}"
iss_suff="it_${INIT_STEPSIZE}"
t_suff="t_$TIMESTEPS"
  
NAME=${PROPOSAL}_proposal_his_model_${lt_suff}_${ss_suff}_${it_suff}_${iss_suff}_${squash_suff}_${t_suff}${NAME_SUFFIX}

LOGDIR=$LOGDIR/$DATASET/$NAME
TEXT_OUTDIR=logdir/$NAME

CUDA_VISIBLE_DEVICES=$GPU python3 mnist.py \
  --logdir=$LOGDIR  \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=his \
  --mode=train \
  --learning_rate=1e-3 \
  --decay_lr \
  --anneal_kl_step=100000 \
  --learn_his_temps=$LEARN_TEMP \
  --learn_his_stepsize=$LEARN_STEPSIZE \
  --his_init_alpha=$INIT_TEMP \
  --his_init_stepsize=$INIT_STEPSIZE \
  --his_T=$TIMESTEPS \
  --squash=$SQUASH \
  --latent_dim=$LATENT_DIM \
  --reparam_vae_prior=true \
  --batch_size=128 \
  --max_steps=10000000 >> ${TEXT_OUTDIR}_train.out 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU python3 mnist.py \
  --logdir=$LOGDIR \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=his \
  --mode=eval \
  --latent_dim=$LATENT_DIM \
  --his_T=$TIMESTEPS \
  --squash=$SQUASH \
  --batch_size=32 \
  --max_steps=10000000 \
  --split=train,valid,test \
  --num_iwae_samples=1,1,1000 >> ${TEXT_OUTDIR}_eval.out 2>&1 &

wait
