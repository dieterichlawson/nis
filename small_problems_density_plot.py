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

NINE_GAUSSIANS_DIST = "nine_gaussians"
TWO_RINGS_DIST = "two_rings"
CHECKERBOARD_DIST = "checkerboard"
TARGET_DISTS = [NINE_GAUSSIANS_DIST, TWO_RINGS_DIST, CHECKERBOARD_DIST]

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_enum("algo", "lars", ["lars","nis", "his"],
                         "The algorithm to run.")
tf.app.flags.DEFINE_enum("target", NINE_GAUSSIANS_DIST,  TARGET_DISTS,
                                 "Distribution to draw data from.")
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
tf.app.flags.DEFINE_integer("K", 128,
                            "The number of samples for NIS and LARS.")
tf.app.flags.DEFINE_integer("num_bins", 500,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_integer("num_samples", 10000000,
                            "Number of samples to use when plotting density.")
tf.app.flags.DEFINE_integer("batch_size", 100000,
                            "The batch size.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                            "Directory for summaries and checkpoints.")
FLAGS = tf.app.flags.FLAGS

def make_sample_density_summary(
        session, 
        data, 
        max_samples_per_batch=100000, 
        num_samples=1000000, 
        num_bins=100):
  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == CHECKERBOARD_DIST:
    bounds = (0,1)
  num_batches = math.ceil(num_samples / float(max_samples_per_batch))
  hist = None
  for i in range(num_batches):
    tf.logging.info("Processing batch %d / %d of samples for density image." % (i+1, num_batches))
    s = session.run(data)
    if hist is None:
      hist = np.histogram2d(s[:,0], s[:,1], bins=num_bins, range=[bounds, bounds])[0]
    else:
      hist += np.histogram2d(s[:,0], s[:,1], bins=num_bins, range=[bounds, bounds])[0]
    np.save(os.path.join(FLAGS.logdir, "density"), hist)
  tf.logging.info("Density image saved to %s" %  os.path.join(FLAGS.logdir, "density.npy"))

def reduce_logavgexp(input_tensor, axis=None, keepdims=None, name=None):
  dims = tf.shape(input_tensor)
  if axis is not None:
    dims = tf.gather(dims, axis)
  denominator = tf.reduce_prod(dims)
  return (tf.reduce_logsumexp(input_tensor,
                              axis=axis,
                              keepdims=keepdims,
                              name=name) - tf.log(tf.to_float(denominator)))

def make_lars_density_summary(
        model,
        num_bins=100):
  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == CHECKERBOARD_DIST:
    bounds = (0,1)

  x = tf.range(bounds[0], bounds[1], delta=(bounds[1]-bounds[0])/float(num_bins))
  X, Y = tf.meshgrid(x, x)
  XY = tf.stack([X,Y], axis=-1)

  log_z = tf.squeeze(model.accept_fn(XY)) + model.proposal.log_prob(XY)
  log_Z = reduce_logavgexp(log_z)
  z = tf.exp(log_z - log_Z)

  plot = tf.reshape(z, [num_bins, num_bins])
  return plot

def main(unused_argv):
  g = tf.Graph()
  with g.as_default():
    energy_fn_layers = [int(x.strip()) for x in FLAGS.energy_fn_sizes.split(",")]
    if FLAGS.algo == "lars":
      tf.logging.info("Running LARS")
      model = lars.SimpleLARS(
          K=FLAGS.K,
          data_dim=2,
          accept_fn_layers=energy_fn_layers)
      plot = make_lars_density_summary(model, num_bins=FLAGS.num_bins)
      with tf.train.SingularMonitoredSession(
        checkpoint_dir=FLAGS.logdir) as sess:
        plot = sess.run(plot)
        np.save(os.path.join(FLAGS.logdir, "density"), plot)
    else:
      if FLAGS.algo == "nis":
        tf.logging.info("Running NIS")
        model = nis.NIS(K=FLAGS.K,
                        data_dim=2,
                        energy_hidden_sizes=energy_fn_layers)
      elif FLAGS.algo == "his":
        tf.logging.info("Running HIS")
        model = his.HIS(FLAGS.his_t,
                        data_dim=2,
                        energy_hidden_sizes=energy_fn_layers,
                        init_step_size=FLAGS.his_stepsize,
                        learn_stepsize=FLAGS.his_learn_stepsize,
                        init_alpha=FLAGS.his_alpha,
                        learn_temps=FLAGS.his_learn_alpha,
                        q_hidden_sizes=energy_fn_layers)
      samples = model.sample([FLAGS.eval_batch_size])
      with tf.train.SingularMonitoredSession(
        checkpoint_dir=FLAGS.logdir) as sess:
        make_sample_density_summary(sess, samples,
            max_samples_per_batch=FLAGS.batch_size,
            num_samples=FLAGS.num_samples,
            num_bins=FLAGS.num_bins)

if __name__ == "__main__":
  tf.app.run(main)
