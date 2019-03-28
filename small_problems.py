import math
import numpy as np
from matplotlib import cm
import functools

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from models.base import mlp
from models import his
from models import nis


NINE_GAUSSIANS_DIST = "nine_gaussians"
TWO_RINGS_DIST = "two_rings"
CHECKERBOARD_DIST = "checkerboard"
TARGET_DISTS = [NINE_GAUSSIANS_DIST, TWO_RINGS_DIST, CHECKERBOARD_DIST]

tf.logging.set_verbosity(tf.logging.INFO)
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
tf.app.flags.DEFINE_float("learning_rate", 1e-4,
                           "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_integer("eval_batch_size", 1000,
                            "The number of examples per eval batch.")
tf.app.flags.DEFINE_integer("density_num_points", 50,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                            "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS

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

def make_lars_loss(target_dist,
                   batch_size=16,
                   Z_batch_size=1024,
                   accept_fn_layers=[10, 10], 
                   log_Z_ema_decay=0.99,
                   Z_ema=None,
                   dtype=tf.float32):
  # Create proposal as standard 2-D Gaussian
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
                                        scale_diag=tf.ones([2], dtype=dtype))
  
  # Sample from target dist (multi-modal Gaussian)
  z_r = target_dist.sample(batch_size)
  
  mlp_fn = functools.partial(
             mlp,
             layer_sizes=accept_fn_layers + [1],
             final_activation=tf.math.log_sigmoid,
             name="a")

  # Compute log a(z), log pi(z), and log q(z)
  log_a_z_r = tf.reshape(mlp_fn(z_r), [batch_size]) # [batch_size]
  log_pi_z_r = proposal.log_prob(z_r) # [batch_size]
  log_q_z_r = target_dist.log_prob(z_r) # [batch_size]

  tf.summary.histogram("log_energy_data", log_a_z_r)
  tf.summary.scalar("min_log_energy_data", tf.reduce_min(log_a_z_r))
  tf.summary.scalar("max_log_energy_data", tf.reduce_max(log_a_z_r))

  # Sample zs from proposal to estimate Z
  z_s = proposal.sample(Z_batch_size) # [Z_batch_size, 2]
  # Compute log a(z) for zs sampled from proposal
  log_a_z_s = tf.reshape(mlp_fn(z_s), [Z_batch_size]) # [Z_batch_size]
  log_ZS = tf.reduce_logsumexp(log_a_z_s) # []

  tf.summary.histogram("log_energy_proposal", log_a_z_s)
  tf.summary.scalar("min_log_energy_proposal", tf.reduce_min(log_a_z_s))
  tf.summary.scalar("max_log_energy_proposal", tf.reduce_max(log_a_z_s))

  if FLAGS.lars_allow_eval_target:
    log_q_z_r = target_dist.log_prob(z_r) #[batch_size]
    # Compute estimate of log Z using importance-weighted samples from minibatch
    iw_log_a_z_r = tf.stop_gradient(log_pi_z_r - log_q_z_r) + log_a_z_r 
    log_Z_curr = tf.reduce_logsumexp([tf.ones_like(iw_log_a_z_r)*log_ZS, iw_log_a_z_r], axis=0)
    log_Z_curr -= tf.log(tf.to_float(Z_batch_size+1))
    log_Z_curr_avg = reduce_logavgexp(log_Z_curr, axis=0) #[]
  else:
    log_Z_curr_avg = log_ZS - tf.log(tf.to_float(Z_batch_size))
  
  # Set up EMA of log_Z
  log_Z_ema = tf.train.ExponentialMovingAverage(decay=log_Z_ema_decay)
  log_Z_curr_avg_sg = tf.stop_gradient(log_Z_curr_avg)
  maintain_log_Z_ema_op = log_Z_ema.apply([log_Z_curr_avg_sg])
  
  # In forward pass, log Z is the smoothed ema version of log Z
  # In backward pass it is the current estimate of log Z, log_Z_curr_avg
  log_Z = log_Z_curr_avg + tf.stop_gradient(log_Z_ema.average(log_Z_curr_avg_sg) - log_Z_curr_avg)
  
  loss = -(log_pi_z_r + log_a_z_r - log_Z[tf.newaxis]) # [batch_size]

  tf.summary.scalar("log_Z_ema", log_Z_ema.average(log_Z_curr_avg_sg))
  return tf.reduce_mean(loss), maintain_log_Z_ema_op, mlp_fn

def make_lars_graph(target_dist,
                    batch_size=16,
                    eval_batch_size=1000,
                    lr=1e-4, 
                    mlp_layers=[10, 10], 
                    dtype=tf.float32):
  loss, ema_op, mlp = make_lars_loss(target_dist, 
                                     batch_size=batch_size, 
                                     accept_fn_layers=mlp_layers,
                                     dtype=dtype)
  eval_loss, eval_ema_op, _ = make_lars_loss(target_dist, 
                                             batch_size=eval_batch_size, 
                                             accept_fn_layers=mlp_layers,
                                             dtype=dtype)
   
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(loss)
  with tf.control_dependencies([ema_op, eval_ema_op]):
    apply_grads_op = opt.apply_gradients(grads, global_step=global_step)
  # Create summaries.

  def f(x):
    e = mlp(x)
    e -= tf.reduce_min(e)
    return tf.exp(e)

  density_image_summary(mlp=f, 
                        num_points=FLAGS.density_num_points, 
                        dtype=dtype)
  tf.summary.scalar("elbo", -loss)
  tf.summary.scalar("eval_elbo", -eval_loss)
  return loss, apply_grads_op, global_step

def density_image_summary(mlp, num_points=50, dtype=tf.float32):
  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == CHECKERBOARD_DIST:
    bounds = (0,1)

  x = tf.range(bounds[0], bounds[1], delta=(bounds[1] - bounds[0])/float(num_points))
  x_dim = tf.shape(x)[0]
  X, Y = tf.meshgrid(x, x)
  z = tf.transpose(tf.reshape(tf.stack([X,Y], axis=0), [2,-1]))
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
                                        scale_diag=tf.ones([2], dtype=dtype))
  
  pi_z = proposal.prob(z)
  energy_z = tf.squeeze(mlp(z))

  unnorm_density = tf.reshape(pi_z*energy_z, [x_dim, x_dim])
  
  def normalize(x):
    no_inf = tf.where(tf.is_inf(x), tf.zeros_like(x), x)
    ma = tf.reduce_max(no_inf)
    mi = tf.reduce_min(no_inf)
    no_inf_x = tf.where(tf.is_inf(x), tf.ones_like(x)*ma, x)
    normalized = tf.clip_by_value((no_inf_x - mi)/(ma-mi), 0., 1.)
    return normalized

  tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])
  
  density_plot = tf_viridis(normalize(unnorm_density))
  energy_fn_plot = tf_viridis(normalize(tf.reshape(energy_z, [x_dim, x_dim])))
  log_energy_fn_plot = tf_viridis(normalize(tf.reshape(tf.log(energy_z), [x_dim, x_dim])))
  
  tf.summary.image("density", density_plot, max_outputs=1, collections=["infrequent_summaries"])
  #tf.summary.image("energy_fn", energy_fn_plot, max_outputs=1, collections=["infrequent_summaries"])
  tf.summary.image("log_energy_fn", log_energy_fn_plot, max_outputs=1, collections=["infrequent_summaries"])


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
  eval_elbo = tf.reduce_mean(model.log_prob(eval_batch))

  tf.summary.scalar("elbo", train_elbo)
  tf.summary.scalar("eval_elbo", eval_elbo, collections=["infrequent_summaries"])

  def f(x):
    e = model.energy_fn(x)
    e -= tf.reduce_min(e)
    return tf.exp(e)

  density_image_summary(mlp=f,
                        num_points=FLAGS.density_num_points)
  
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-train_elbo)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return train_elbo, train_op, global_step

def his_density_image_summary(his_model, num_points=50):
  if FLAGS.target == NINE_GAUSSIANS_DIST or FLAGS.target == TWO_RINGS_DIST:
    bounds = (-2,2)
  elif FLAGS.target == CHECKERBOARD_DIST:
    bounds = (0,1)

  x = tf.range(bounds[0], bounds[1], delta=(bounds[1]-bounds[0])/float(num_points))
  X, Y = tf.meshgrid(x, x)
  Z = tf.stack([X,Y], axis=-1)
  unnorm_density = tf.exp(his_model.log_prob(Z, num_samples=100))

  def normalize(x):
    no_inf = tf.where(tf.is_inf(x), tf.zeros_like(x), x)
    ma = tf.reduce_max(no_inf)
    mi = tf.reduce_min(no_inf)
    no_inf_x = tf.where(tf.is_inf(x), tf.ones_like(x)*ma, x)
    normalized = tf.clip_by_value((no_inf_x - mi)/(ma-mi), 0., 1.)
    return normalized

  tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])
  
  plot = tf_viridis(normalize(unnorm_density))
  tf.summary.image("density", plot, max_outputs=1, collections=["infrequent_summaries"])
  
  log_energy_fn = tf.squeeze(his_model.energy_fn(Z))
  energy_fn = tf.exp(log_energy_fn)

  tf.summary.image("energy_fn",
          tf_viridis(normalize(energy_fn)),
          max_outputs=1,
          collections=["infrequent_summaries"])
  tf.summary.image("log_energy_fn",
          tf_viridis(normalize(log_energy_fn)),
          max_outputs=1,
          collections=["infrequent_summaries"])

def make_his_graph(target_dist,
                   batch_size=16,
                   eval_batch_size=1000,
                   T=100,
                   step_size=1e-2,
                   lr=1e-4,
                   mlp_layers=[10,10],
                   dtype=tf.float32):
  data = target_dist.sample(batch_size)
  eval_data = target_dist.sample(eval_batch_size)
  model = his.HIS(T,
                  data_dim=2,
                  energy_hidden_sizes=mlp_layers,
                  init_step_size=step_size,
                  q_hidden_sizes=mlp_layers)
  elbo = model.log_prob(data)
  elbo = tf.reduce_mean(elbo)
  eval_elbo = tf.reduce_mean(model.log_prob(eval_data))
  tf.summary.scalar("elbo", elbo)
  tf.summary.scalar("eval_elbo", eval_elbo, collections=["infrequent_summaries"])
  his_density_image_summary(model, num_points=FLAGS.density_num_points)
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
        save_steps=500,
        output_dir=FLAGS.logdir,
        summary_op=tf.summary.merge_all(key="infrequent_summaries")
    )
    hooks.append(infrequent_summary_hook)
  return hooks

def main(unused_argv):
  g = tf.Graph()
  with g.as_default():
    target = get_target_distribution(FLAGS.target)
    energy_fn_layers = [int(x.strip()) for x in FLAGS.energy_fn_sizes.split(",")]
    if FLAGS.algo == "lars":
      print("Running LARS")
      loss, train_op, global_step = make_lars_graph(
        target_dist=target,
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
        K=128,
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
        step_size=FLAGS.his_stepsize,
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
