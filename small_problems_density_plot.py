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
tf.app.flags.DEFINE_integer("eval_batch_size", 1000,
                            "The number of examples per eval batch.")
tf.app.flags.DEFINE_integer("K", 128,
                            "The number of samples for NIS and LARS.")
tf.app.flags.DEFINE_integer("density_num_points", 100,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_integer("density_num_samples", 1000000,
                            "Number of samples to use when plotting density.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                            "Directory for summaries and checkpoints.")
FLAGS = tf.app.flags.FLAGS

def make_sample_density_summary(
        session, 
        data, 
        title, 
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
    elif FLAGS.algo == "nis":
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
      make_sample_density_summary(sess, samples, "density", 
              max_samples_per_batch=FLAGS.eval_batch_size,
              num_samples=FLAGS.density_num_samples, 
              num_bins=FLAGS.density_num_points)

if __name__ == "__main__":
  tf.app.run(main)
