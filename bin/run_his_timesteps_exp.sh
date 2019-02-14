sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=100 \
  --gpu=0 &

sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=50 \
  --gpu=1 &

sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=10 \
  --gpu=2 &

sh bin/run_his_experiment.sh \
  --dataset=raw_mnist \
  --proposal=gaussian_vae \
  --init_temp=0.9995 \
  --init_stepsize=0.01 \
  --timsteps=250 \
  --gpu=3 &
