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
tf.app.flags.DEFINE_enum("mode", "train", ["train", "density_img"], "Mode to run.")
tf.app.flags.DEFINE_enum("algo", "lars", ["lars","nis", "his"],
                         "The algorithm to run.")
tf.app.flags.DEFINE_boolean("lars_allow_eval_target", False,
                            "Whether LARS is allowed to evaluate the target density.")
tf.app.flags.DEFINE_enum("target", NINE_GAUSSIANS_DIST,  TARGET_DISTS,
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
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS

tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])

class Ring2D(tfd.Distribution):
  
  def __init__(self,
               radius_dist=None,
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True,
               name="Ring"):
    parameters = dict(locals())
    loc = tf.zeros([2], dtype=dtype)
    if radius_dist is None:
      radius_dist = tfd.Normal(loc=1., scale=0.1)
    self._loc = loc
    self._radius_dist= radius_dist
    super(Ring2D, self).__init__(
        dtype=dtype,
        reparameterization_type=tfd.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc],
        name=name)

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc
  
  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
      tf.shape(self._loc)[:-1],
      self._radius_dist.batch_shape_tensor)

  def _batch_shape(self):
    return tf.broadcast_static_shape(
      self._loc.get_shape()[:-1],
      self._radius_dist.batch_shape)
  
  def _event_shape_tensor(self):
    return tf.constant([2], dtype=dtypes.int32)

  def _event_shape(self):
    return tf.TensorShape([2])
  
  def _sample_n(self, n, seed=None):
    new_shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    thetas = tf.random_uniform(
        new_shape, seed=seed, dtype=self.dtype)*2.*math.pi
    rs = self._radius_dist.sample(new_shape, seed=seed)
    vecs = tf.stack([tf.math.sin(thetas), tf.math.cos(thetas)], axis=-1)
    
    sample = vecs*tf.expand_dims(rs, axis=-1)
    return tf.cast(sample, self.dtype)

  def _log_prob(self, event):
    radii = tf.norm(event, axis=-1, ord=2)
    return self._radius_dist.log_prob(radii) - tf.log(2*math.pi*radii)

def two_rings_dist(scale=0.1):
  r_dist = tfd.Mixture(
    cat=tfd.Categorical(probs=[1., 1.]),
    components=[tfd.Normal(loc=0.6, scale=scale), tfd.Normal(loc=1.3,scale=scale)])
  return Ring2D(radius_dist=r_dist)
 
def checkerboard_dist(num_splits=4):
  bounds = np.linspace(0., 1., num=(num_splits+1), endpoint=True)
  uniforms = []
  for i in range(num_splits):
    for j in range(num_splits):
      if ((i % 2 == 0 and j % 2 == 0) or
          (i % 2 != 0 and j % 2 != 0)):
        low = tf.convert_to_tensor([bounds[i], bounds[j]], dtype=tf.float32)
        high = tf.convert_to_tensor([bounds[i+1], bounds[j+1]], dtype=tf.float32)
        u = tfd.Uniform(low=low, high=high)
        u = tfd.Independent(u, reinterpreted_batch_ndims=1)
        uniforms.append(u)
  return tfd.Mixture(
    cat=tfd.Categorical(probs=[1.]*len(uniforms)),
    components=uniforms)

def nine_gaussians_dist(variance=0.1):
  """Creates a mixture of 9 2-D gaussians on a 3x3 grid centered at 0."""
  components = []
  for i in [-1., 0. , 1.]:
    for j in [-1., 0. , 1.]:
      loc = tf.constant([i,j], dtype=tf.float32)
      scale = tf.ones_like(loc)*tf.sqrt(variance)
      components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale))
  return tfd.Mixture(
    cat=tfd.Categorical(probs=tf.ones([9], dtype=tf.float32)/9.),
    components=components)

def get_target_distribution(name):
  if name == NINE_GAUSSIANS_DIST:
    return nine_gaussians_dist(variance=FLAGS.nine_gaussians_variance)
  elif name == TWO_RINGS_DIST:
    return two_rings_dist()
  elif name == CHECKERBOARD_DIST:
    return checkerboard_dist()
  else:
    raise ValueError("Invalid target name.")

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
                          FLAGS.density_num_points,
                          'energy/lars')
  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_points)


  tf.summary.scalar("elbo", tf.reduce_mean(log_p))
  tf.summary.scalar("eval_elbo", tf.reduce_mean(eval_log_p))
  return -tf.reduce_mean(log_p), apply_grads_op, global_step

#def density_image_summary(mlp, num_points=50, dtype=tf.float32):
#  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
#    bounds = (-2,2)
#  elif FLAGS.target == CHECKERBOARD_DIST:
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
                        FLAGS.density_num_points,
                        'energy/nis')
  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_points)
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-train_elbo)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return train_elbo, train_op, global_step

def density_image_summary(log_density, num_points, title):
  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == CHECKERBOARD_DIST:
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
  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == CHECKERBOARD_DIST:
    bounds = (0,1)
  data = model.sample([num_samples])

  #def log_gaussian_kde(data, eval_data):
  #  kernel = gaussian_kde(data.T)
  #  eval_data = np.reshape(eval_data, [FLAGS.density_num_points * FLAGS.density_num_points, -1]).T
  #  return np.reshape(kernel.logpdf(eval_data), [FLAGS.density_num_points, -1])
  #tf_log_gaussian_kde = lambda x: tf.to_float(tf.py_func(
  #    log_gaussian_kde, [data, x], [tf.float64]))

  #density_image_summary(tf_log_gaussian_kde, FLAGS.density_num_points, title)

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
                        FLAGS.density_num_points,
                        'energy/hamiltonian_potential')
  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_points)
  #his_density_image_summary(model, num_points=FLAGS.density_num_points)
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

def run_train():
  g = tf.Graph()
  with g.as_default():
    target = get_target_distribution(FLAGS.target)
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

def make_lars_density_summary_graph(session, K=1024, num_points=200, 
        num_samples=10000000, mlp_layers=[10,10], dtype=tf.float32):
  model = lars.SimpleLARS(
            K=K,
            data_dim=2,
            accept_fn_layers=mlp_layers,
            dtype=dtype)

  sample_image_summary(model, 'density', num_samples=FLAGS.density_num_samples,
          num_bins=FLAGS.density_num_points)


  tf.summary.scalar("elbo", tf.reduce_mean(log_p))
  tf.summary.scalar("eval_elbo", tf.reduce_mean(eval_log_p))
  return -tf.reduce_mean(log_p), apply_grads_op, global_step

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

def run_density():
  g = tf.Graph()
  with g.as_default():
    target = get_target_distribution(FLAGS.target)
    energy_fn_layers = [int(x.strip()) for x in FLAGS.energy_fn_sizes.split(",")]
    if FLAGS.algo == "lars":
      print("Running LARS")
      model = lars.SimpleLARS(
          K=FLAGS.K,
          data_dim=2,
          accept_fn_layers=energy_fn_layers)
    elif FLAGS.algo == "nis":
      print("Running NIS")
      model = nis.NIS(K=FLAGS.K,
                data_dim=2,
                energy_hidden_sizes=energy_fn_layers)
    elif FLAGS.algo == "his":
      print("Running HIS")
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

def main(unused_argv):
  if FLAGS.mode == "train":
    run_train()
  elif FLAGS.mode == "density_img":
    run_density()

if __name__ == "__main__":
  tf.app.run(main)
