#!/bin/bash
#SBATCH --exclude=lion7
#SBATCH --job-name=vae_prior_nis_model_3
#SBATCH --output=slurm_logs/vae_prior_nis_model_3_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres gpu:1080ti:2
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --constraint=gpu_12gb
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=6000

trap "exit" INT TERM
trap "kill 0" EXIT

#module purge
#module load cuda-9.0

DATASET=static_mnist
PROPOSAL=bernoulli_vae
MODEL=nis
LATENT_DIM=50
NAME=${PROPOSAL}_proposal_${MODEL}_model
LOGDIR=/tmp/experiments/$DATASET/$NAME

CUDA_VISIBLE_DEVICES=0 python3 mnist.py \
  --logdir=$LOGDIR  \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=$MODEL \
  --mode=train \
  --learning_rate=3e-4 \
  --decay_lr \
  --anneal_kl_step=100000 \
  --latent_dim=$LATENT_DIM \
  --reparam_vae_prior=false \
  --batch_size=128 \
  --max_steps=10000000 &

CUDA_VISIBLE_DEVICES=1 python3 mnist.py \
  --logdir=$LOGDIR \
  --dataset=$DATASET \
  --proposal=$PROPOSAL \
  --model=$MODEL \
  --mode=eval \
  --latent_dim=$LATENT_DIM \
  --batch_size=6 \
  --max_steps=10000000 \
  --split=train,valid,test \
  --num_iwae_samples=1,1,1000 &

wait


