#!/bin/bash

sh bin/run_experiment -d raw_mnist \
                      -p gaussian_vae \
                      -m nis \
                      -l /tmp/experiments/continuous_grad_exp \
                      -g 0 \
                      -r &

sh bin/run_experiment -d raw_mnist \
                      -p gaussian_vae \
                      -m nis \
                      -l /tmp/experiments/continuous_grad_exp \
                      -g 1 &
