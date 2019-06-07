import math
import os
import numpy as np
from matplotlib import cm
from scipy.stats import gaussian_kde
import functools

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from models.base import mlp
from models import his
from models import nis
from models import lars
import small_problems_dists as dists

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_enum("algo", "lars", ["lars","nis", "his"],
                         "The algorithm to run.")
tf.app.flags.DEFINE_boolean("lars_allow_eval_target", False,
                            "Whether LARS is allowed to evaluate the target density.")
tf.app.flags.DEFINE_enum("target", dists.NINE_GAUSSIANS_DIST,  dists.TARGET_DISTS,
                         "Distribution to draw data from.")
tf.app.flags.DEFINE_float("nine_gaussians_variance", 0.1,
                          "Variance for the mixture components in the nine gaussians.")
tf.app.flags.DEFINE_string("energy_fn_sizes", "20,20",
                           "List of hidden layer sizes for energy function as as comma "
                            "separated list.")
tf.app.flags.DEFINE_integer("his_t", 25, 
                            "Number of steps for hamiltonian importance sampling.")
tf.app.flags.DEFINE_float("his_stepsize", 1e-2, 
                            "Stepsize for hamiltonian importance sampling.")
tf.app.flags.DEFINE_float("his_alpha", 0.995, 
                            "Alpha for hamiltonian importance sampling.")
tf.app.flags.DEFINE_boolean("his_learn_stepsize", False,
                            "Allow HIS to learn the stepsize")
tf.app.flags.DEFINE_boolean("his_learn_alpha", False,
                            "Allow HIS to learn alpha.")
tf.app.flags.DEFINE_float("learning_rate", 1e-4,
                           "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_integer("density_num_bins", 100,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_integer("density_num_samples", 100000,
                            "Number of samples to use when plotting density.")
tf.app.flags.DEFINE_integer("eval_batch_size", 1000,
                            "The number of examples per eval batch.")
tf.app.flags.DEFINE_integer("K", 128,
                            "The number of samples for NIS and LARS.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                            "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS

tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])

def reduce_logavgexp(input_tensor, axis=None, keepdims=None, name=None):
  dims = tf.shape(input_tensor)
  if axis is not None:
    dims = tf.gather(dims, axis)
  denominator = tf.reduce_prod(dims)
  return (tf.reduce_logsumexp(input_tensor, 
                              axis=axis, 
                              keepdims=keepdims, 
                              name=name) - tf.log(tf.to_float(denominator)))

def make_lars_graph(target_dist,
                    K=100,
                    batch_size=16,
                    eval_batch_size=1000,
                    lr=1e-4, 
                    mlp_layers=[10, 10], 
                    dtype=tf.float32):

  model = lars.SimpleLARS(
            K=K,
            data_dim=2,
            accept_fn_layers=mlp_layers,
            dtype=dtype)

  train_data = target_dist.sample(batch_size)
  log_p, ema_op = model.log_prob(train_data)
  test_data = target_dist.sample(eval_batch_size)
  eval_log_p, eval_ema_op = model.log_prob(test_data)

  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(-tf.reduce_mean(log_p))
  with tf.control_dependencies([ema_op, eval_ema_op]):
    apply_grads_op = opt.apply_gradients(grads, global_step=global_step)

  density_image_summary(lambda x: tf.squeeze(model.accept_fn(x))
                          + model.proposal.log_prob(x),
                          FLAGS.density_num_bins,
                          'energy/lars')
  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_bins)


  tf.summary.scalar("elbo", tf.reduce_mean(log_p))
  tf.summary.scalar("eval_elbo", tf.reduce_mean(eval_log_p))
  return -tf.reduce_mean(log_p), apply_grads_op, global_step

#def density_image_summary(mlp, num_points=50, dtype=tf.float32):
#  if FLAGS.target == dists.NINE_GAUSSIANS_DIST or FLAGS.target == dists.TWO_RINGS_DIST:
#    bounds = (-2,2)
#  elif FLAGS.target == dists.CHECKERBOARD_DIST:
#    bounds = (0,1)
#
#  x = tf.range(bounds[0], bounds[1], delta=(bounds[1] - bounds[0])/float(num_points))
#  x_dim = tf.shape(x)[0]
#  X, Y = tf.meshgrid(x, x)
#  z = tf.transpose(tf.reshape(tf.stack([X,Y], axis=0), [2,-1]))
#  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
#                                        scale_diag=tf.ones([2], dtype=dtype))
#  
#  pi_z = proposal.prob(z)
#  energy_z = tf.squeeze(mlp(z))
#
#  unnorm_density = tf.reshape(pi_z*energy_z, [x_dim, x_dim])
#  
#  def normalize(x):
#    no_inf = tf.where(tf.is_inf(x), tf.zeros_like(x), x)
#    ma = tf.reduce_max(no_inf)
#    mi = tf.reduce_min(no_inf)
#    no_inf_x = tf.where(tf.is_inf(x), tf.ones_like(x)*ma, x)
#    normalized = tf.clip_by_value((no_inf_x - mi)/(ma-mi), 0., 1.)
#    return normalized
#
#  tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])
#  
#  density_plot = tf_viridis(normalize(unnorm_density))
#  energy_fn_plot = tf_viridis(normalize(tf.reshape(energy_z, [x_dim, x_dim])))
#  log_energy_fn_plot = tf_viridis(normalize(tf.reshape(tf.log(energy_z), [x_dim, x_dim])))
#  
#  tf.summary.image("density", density_plot, max_outputs=1, collections=["infrequent_summaries"])
#  #tf.summary.image("energy_fn", energy_fn_plot, max_outputs=1, collections=["infrequent_summaries"])
#  tf.summary.image("log_energy_fn", log_energy_fn_plot, max_outputs=1, collections=["infrequent_summaries"])


# Code for NIS model
def make_nis_graph(target_dist,
                   batch_size=16,
                   eval_batch_size=1000,
                   K=100,
                   lr=1e-4,
                   mlp_layers=[10,10],
                   dtype=tf.float32):
  train_batch = target_dist.sample(batch_size)
  eval_batch = target_dist.sample(eval_batch_size)

  model = nis.NIS(K=K,
                  data_dim=2,
                  energy_hidden_sizes=mlp_layers)

  train_elbo = tf.reduce_mean(model.log_prob(train_batch))
  eval_elbo = tf.reduce_mean(model.log_prob(eval_batch, num_samples=1000))

  tf.summary.scalar("elbo", train_elbo)
  tf.summary.scalar("eval_elbo", eval_elbo, collections=["infrequent_summaries"])
  density_image_summary(lambda x: tf.squeeze(model.energy_fn(x))
                        + model.proposal.log_prob(x),
                        FLAGS.density_num_bins,
                        'energy/nis')
  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_bins)
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-train_elbo)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return train_elbo, train_op, global_step

def density_image_summary(log_density, num_points, title):
  if FLAGS.target == dists.NINE_GAUSSIANS_DIST or FLAGS.target == dists.TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == dists.CHECKERBOARD_DIST:
    bounds = (0,1)

  x = tf.range(bounds[0], bounds[1], delta=(bounds[1]-bounds[0])/float(num_points))
  X, Y = tf.meshgrid(x, x)
  XY = tf.stack([X,Y], axis=-1)

  log_z = log_density(XY)
  log_Z = reduce_logavgexp(log_z)
  z = tf.exp(log_z - log_Z)

  plot = tf.reshape(z, [1, num_points, num_points, 1]) #tf_viridis(z)
  tf.summary.image(title, plot, max_outputs=1, collections=["infrequent_summaries"])

def sample_image_summary(model, title, num_samples=100000, num_bins=50):
  if FLAGS.target == dists.NINE_GAUSSIANS_DIST or FLAGS.target == dists.TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == dists.CHECKERBOARD_DIST:
    bounds = (0,1)
  data = model.sample([num_samples])

  #def log_gaussian_kde(data, eval_data):
  #  kernel = gaussian_kde(data.T)
  #  eval_data = np.reshape(eval_data, [FLAGS.density_num_bins * FLAGS.density_num_bins, -1]).T
  #  return np.reshape(kernel.logpdf(eval_data), [FLAGS.density_num_bins, -1])
  #tf_log_gaussian_kde = lambda x: tf.to_float(tf.py_func(
  #    log_gaussian_kde, [data, x], [tf.float64]))

  #density_image_summary(tf_log_gaussian_kde, FLAGS.density_num_bins, title)

  def _hist2d(x, y):
    return np.histogram2d(x, y, bins=num_bins, range=[bounds,bounds])[0]

  tf_hist2d = lambda x, y: tf.py_func(_hist2d, [x, y], [tf.float64])
  plot = tf.expand_dims(tf_hist2d(data[:, 0], data[:, 1]), -1)
  tf.summary.image(title, plot, max_outputs=1, collections=["infrequent_summaries"])

def make_his_graph(target_dist,
                   batch_size=16,
                   eval_batch_size=1000,
                   T=100,
                   stepsize=1e-2,
                   alpha=0.995,
                   learn_stepsize=False,
                   learn_alpha=False,
                   lr=1e-4,
                   mlp_layers=[10,10],
                   dtype=tf.float32):
  data = target_dist.sample(batch_size)
  eval_data = target_dist.sample(eval_batch_size)
  model = his.HIS(T,
                  data_dim=2,
                  energy_hidden_sizes=mlp_layers,
                  init_step_size=stepsize,
                  learn_stepsize=learn_stepsize,
                  init_alpha=alpha,
                  learn_temps=learn_alpha,
                  q_hidden_sizes=mlp_layers)
  elbo = model.log_prob(data)
  elbo = tf.reduce_mean(elbo)
  eval_elbo = tf.reduce_mean(model.log_prob(eval_data, num_samples=1000))
  tf.summary.scalar("elbo", elbo)
  tf.summary.scalar("eval_elbo", eval_elbo, collections=["infrequent_summaries"])
  density_image_summary(lambda x: -model.hamiltonian_potential(x),
                        FLAGS.density_num_bins,
                        'energy/hamiltonian_potential')
  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_bins)
  #his_density_image_summary(model, num_points=FLAGS.density_num_bins)
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-elbo)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return elbo, train_op, global_step

def make_log_hooks(global_step, loss):
  hooks = []
  def summ_formatter(d):
    return ("Step {step}, loss: {loss:.5f}".format(**d))
  loss_hook = tf.train.LoggingTensorHook(
      {"step": global_step, "loss": loss},
      every_n_iter=FLAGS.summarize_every,
      formatter=summ_formatter)
  hooks.append(loss_hook)
  if len(tf.get_collection("infrequent_summaries")) > 0:
    infrequent_summary_hook = tf.train.SummarySaverHook(
        save_steps=1000,
        output_dir=FLAGS.logdir,
        summary_op=tf.summary.merge_all(key="infrequent_summaries")
    )
    hooks.append(infrequent_summary_hook)
  return hooks

def main(unused_argv):
  g = tf.Graph()
  with g.as_default():
    target = dists.get_target_distribution(FLAGS.target,
            nine_gaussians_variance=FLAGS.nine_gaussians_variance)
    energy_fn_layers = [int(x.strip()) for x in FLAGS.energy_fn_sizes.split(",")]
    if FLAGS.algo == "lars":
      print("Running LARS")
      loss, train_op, global_step = make_lars_graph(
        target_dist=target,
        K=FLAGS.K,
        batch_size=FLAGS.batch_size, 
        eval_batch_size=FLAGS.eval_batch_size,
	lr=FLAGS.learning_rate, 
        mlp_layers=energy_fn_layers,
	dtype=tf.float32)
    elif FLAGS.algo == "nis":
      print("Running NIS")
      loss, train_op, global_step = make_nis_graph(
        target_dist=target,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        K=FLAGS.K,
        lr=FLAGS.learning_rate,
        mlp_layers=energy_fn_layers,
        dtype=tf.float32)
    elif FLAGS.algo == "his":
      print("Running HIS")
      loss, train_op, global_step = make_his_graph(
        target_dist=target,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        T=FLAGS.his_t,
        stepsize=FLAGS.his_stepsize,
        alpha=FLAGS.his_alpha,
        learn_stepsize=FLAGS.his_learn_stepsize,
        learn_alpha=FLAGS.his_learn_alpha,
        lr=FLAGS.learning_rate,
        mlp_layers=energy_fn_layers,
        dtype=tf.float32)

    log_hooks = make_log_hooks(global_step, loss) 
    with tf.train.MonitoredTrainingSession(
        master="",
        is_chief=True,
        hooks=log_hooks,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=120,
        save_summaries_steps=FLAGS.summarize_every,
        log_step_count_steps=FLAGS.summarize_every) as sess:
      cur_step = -1
      while True:
        if sess.should_stop() or cur_step > FLAGS.max_steps:
          break
        # run a step
        _, cur_step = sess.run([train_op, global_step])

if __name__ == "__main__":
  tf.app.run(main)
