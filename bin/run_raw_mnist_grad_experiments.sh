#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

PROBLEM=raw_mnist
BATCH_SIZE=128
LOGDIR=/tmp/experiments
LR=0.0003
TEXT_OUTDIR=logdir

python3 mnist.py --mode=train \
                 --dataset=raw_mnist \
                 --proposal=gaussian \
                 --model=gaussian_vae \
                 --decoder_hidden_sizes=300,300 \
                 --q_hidden_sizes=300,300 \
                 --reparameterize_proposal=true \
                 --vae_decoder_nn_scale=true \
                 --learning_rate=${LR} \
                 --batch_size=${BATCH_SIZE} \
                 --logdir=${LOGDIR}/gaussian_vae_nn_scale >> ${TEXT_OUTDIR}/gaussian_vae_nn_scale.log 2>&1 &

python3 mnist.py --mode=train \
                 --dataset=raw_mnist \
                 --proposal=gaussian \
                 --model=gaussian_vae \
                 --decoder_hidden_sizes=300,300 \
                 --q_hidden_sizes=300,300 \
                 --reparameterize_proposal=true \
                 --vae_decoder_nn_scale=false \
                 --learning_rate=${LR} \
                 --batch_size=${BATCH_SIZE} \
                 --logdir=${LOGDIR}/gaussian_vae_const_scale >> ${TEXT_OUTDIR}/gaussian_vae_const_scale.log 2>&1 &

python3 mnist.py --mode=train \
                 --dataset=raw_mnist \
                 --proposal=gaussian_vae \
                 --vae_decoder_nn_scale=true \
                 --model=nis \
                 --K=1028 \
                 --decoder_hidden_sizes=100,100 \
                 --q_hidden_sizes=100,100 \
                 --energy_hidden_sizes=300,300 \
                 --reparameterize_proposal=true \
                 --learning_rate=${LR} \
                 --batch_size=${BATCH_SIZE} \
                 --logdir=${LOGDIR}/nis_model_vae_nn_scale_prop_with_grads >> ${TEXT_OUTDIR}/nis_model_vae_nn_scale_prop_with_grads.log 2>&1 &
              
python3 mnist.py --mode=train \
                 --dataset=raw_mnist \
                 --proposal=gaussian_vae \
                 --vae_decoder_nn_scale=false \
                 --model=nis \
                 --K=1028 \
                 --decoder_hidden_sizes=100,100 \
                 --q_hidden_sizes=100,100 \
                 --energy_hidden_sizes=300,300 \
                 --reparameterize_proposal=true \
                 --learning_rate=${LR} \
                 --batch_size=${BATCH_SIZE} \
                 --logdir=${LOGDIR}/nis_model_vae_const_scale_prop_with_grads >> ${TEXT_OUTDIR}/nis_model_vae_const_scale_prop_with_grads.log 2>&1 &

python3 mnist.py --mode=train \
                 --dataset=raw_mnist \
                 --proposal=gaussian_vae \
                 --vae_decoder_nn_scale=true \
                 --model=nis \
                 --K=1028 \
                 --decoder_hidden_sizes=100,100 \
                 --q_hidden_sizes=100,100 \
                 --energy_hidden_sizes=300,300 \
                 --reparameterize_proposal=false \
                 --learning_rate=${LR} \
                 --batch_size=${BATCH_SIZE} \
                 --logdir=${LOGDIR}/nis_model_vae_nn_scale_prop_no_grads >> ${TEXT_OUTDIR}/nis_model_vae_nn_scale_prop_no_grads.log 2>&1 &

python3 mnist.py --mode=train \
                 --dataset=raw_mnist \
                 --proposal=gaussian_vae \
                 --vae_decoder_nn_scale=false \
                 --model=nis \
                 --K=1028 \
                 --decoder_hidden_sizes=100,100 \
                 --q_hidden_sizes=100,100 \
                 --energy_hidden_sizes=300,300 \
                 --reparameterize_proposal=false \
                 --learning_rate=${LR} \
                 --batch_size=${BATCH_SIZE} \
                 --logdir=${LOGDIR}/nis_model_vae_const_scale_prop_no_grads >> ${TEXT_OUTDIR}/nis_model_vae_const_scale_prop_no_grads.log 2>&1 &

wait
