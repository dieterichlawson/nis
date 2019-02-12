#!/bin/bash

trap "exit" INT TERM
trap "kill 0" EXIT

DATASET="static_mnist"
LOGDIR=/tmp/experiments/static_mnist
NAME_SUFFIX=""
D="$(dirname "$0")"

# Standard VAE w/ Gaussian prior
sh $D/run_experiment.sh -d ${DATASET} \
                        -p "gaussian" \
                        -m "bernoulli_vae" \
                        -l ${LOGDIR} \
                        -g 0 &

# VAE w/ NIS prior
sh $D/run_experiment.sh -d ${DATASET} \
                        -p "nis" \
                        -m "bernoulli_vae" \
                        -l ${LOGDIR} \
                        -g 1 &

# NIS w/ Bernoulli VAE prior, no grads to prior
sh $D/run_experiment.sh -d ${DATASET} \
                        -p "bernoulli_vae" \
                        -m "nis" \
                        -l ${LOGDIR} \
                        -g 2 &

# NIS w/ Bernoulli VAE prior, including grads to prior
sh $D/run_experiment.sh -d ${DATASET} \
                        -p "bernoulli_vae" \
                        -m "nis" \
                        -l ${LOGDIR} \
                        -g 3 \
                        -r &

wait
