sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --learn_temp \
  --learn_stepsize \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=100 \
  --gpu=0 &

sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --learn_temp \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=100 \
  --gpu=1 &

sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --learn_stepsize \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=100 \
  --gpu=2 &

sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=100 \
  --gpu=3 &
