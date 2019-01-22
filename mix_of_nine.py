import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import functools
import tfmpl

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_enum("algo", "lars", ["lars","nis"],
                         "The algorithm to run.")
tf.app.flags.DEFINE_float("learning_rate", 1e-4,
                           "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                            "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS


def mixture_of_nine(scale=0.1):
  """Creates a mixture of 9 2-D gaussians on a 3x3 grid centered at 0."""
  components = []
  for i in [-1., 0. , 1.]:
    for j in [-1., 0. , 1.]:
      loc = tf.constant([i,j], dtype=tf.float32)
      scale = tf.ones_like(loc)*scale
      components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale))
      
  return tfd.Mixture(
    cat=tfd.Categorical(probs=tf.ones([9], dtype=tf.float32)/9.),
    components=components)


def plot_mixture():
  x = tf.range(-2, 2, delta=0.1)
  X, Y = tf.meshgrid(x, x)
  z = tf.transpose(tf.reshape(tf.stack([X,Y], axis=0), [2,-1]))
  dist = mixture_of_nine()
  probs = dist.prob(z)
  sess = tf.InteractiveSession()
  z_e, probs_e = sess.run([z, probs])
  x = z_e[:,0]
  y = z_e[:,1]
  z = np.reshape(probs_e, [40,40])
  plt.imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), interpolation='none')
  plt.grid(None)
  plt.colorbar()
  plt.show()

def mlp(inputs, layer_sizes, 
        inner_activation=tf.math.tanh, 
        final_activation=tf.math.log_sigmoid,
        name=None):
  """Creates a simple multi-layer perceptron."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    for i, s in enumerate(layer_sizes):
      inputs = tf.layers.dense(inputs, 
                               units=s, 
                               activation=inner_activation, 
                               kernel_initializer=tf.initializers.glorot_uniform,
                               name="layer_%d" % (i+1))
    output = tf.layers.dense(inputs, 
                             units=1, 
                             activation=final_activation,
                             kernel_initializer=tf.initializers.glorot_uniform,
                             name="layer_%d" % (len(layer_sizes)+1))
  return output


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
                   dtype=tf.float32):
  # Create proposal as standard 2-D Gaussian
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
                                        scale_diag=tf.ones([2], dtype=dtype))
  
  # Sample from target dist (multi-modal Gaussian)
  z_r = target_dist.sample(batch_size)
  
  # Compute log a(z), log pi(z), and log q(z)
  log_a_z_r = tf.reshape(mlp(z_r, accept_fn_layers, name="a"), [batch_size]) # [batch_size]
  log_pi_z_r = proposal.log_prob(z_r) # [batch_size]
  log_q_z_r = target_dist.log_prob(z_r) # [batch_size]

  # Sample zs from proposal to estimate Z
  z_s = proposal.sample(Z_batch_size) # [Z_batch_size, 2]
  # Compute log a(z) for zs sampled from proposal
  log_a_z_s = tf.reshape(mlp(z_s, accept_fn_layers, name="a"), [Z_batch_size]) # [Z_batch_size]
  #log_Z = reduce_logavgexp(log_a_z_s) # []
  log_ZS = tf.reduce_logsumexp(log_a_z_s) # []
  
  # Compute estimate of log Z using importance-weighted samples from minibatch
  iw_log_a_z_r = tf.stop_gradient(log_pi_z_r - log_q_z_r) + log_a_z_r 
  log_Z_curr = tf.reduce_logsumexp([tf.ones_like(iw_log_a_z_r)*log_ZS, iw_log_a_z_r], axis=0)
  log_Z_curr -= tf.log(tf.to_float(Z_batch_size+1))
  
  log_Z_curr_avg = reduce_logavgexp(log_Z_curr, axis=0) #[]
  
  # Set up EMA of log_Z
  log_Z_ema = tf.train.ExponentialMovingAverage(decay=log_Z_ema_decay)
  log_Z_curr_avg_sg = tf.stop_gradient(log_Z_curr_avg)
  maintain_log_Z_ema_op = log_Z_ema.apply([log_Z_curr_avg_sg])
  
  # In forward pass, log Z is the smoothed ema version of log Z
  # In backward pass it is the current estimate of log Z, log_Z_curr_avg
  log_Z = log_Z_curr_avg + tf.stop_gradient(log_Z_ema.average(log_Z_curr_avg_sg) - log_Z_curr_avg)
  
  loss = -(log_pi_z_r + log_a_z_r - log_Z[tf.newaxis]) # [batch_size]

  tf.summary.scalar("log Z ema", log_Z_ema.average(log_Z_curr_avg_sg))
  return tf.reduce_mean(loss), maintain_log_Z_ema_op

def make_lars_graph(batch_size=16, 
                    lr=1e-4, 
                    mlp_layers=[10, 10], 
                    dtype=tf.float32):
  target = mixture_of_nine()
  loss, ema_op = make_lars_loss(target, 
                                batch_size=batch_size, 
                                accept_fn_layers=mlp_layers,
                                dtype=dtype)
  
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(loss)
  with tf.control_dependencies([ema_op]):
    apply_grads_op = opt.apply_gradients(grads, global_step=global_step)
  # Create summaries.
  density_image_summary(dtype=dtype, mlp_name="a", mlp_layers=mlp_layers)
  tf.summary.scalar("loss", loss)
  return loss, apply_grads_op, global_step

@tfmpl.figure_tensor
def plot_density(unnorm_density):
  fig = tfmpl.create_figure()
  ax = fig.add_subplot(111)
  ax.imshow(unnorm_density, extent=(-2, 2, -2, 2), interpolation='none')
  ax.grid(False)
  return fig

def density_image_summary(mlp_name="a", 
                          final_activation=tf.math.sigmoid,
                          mlp_layers=[10,10],
                          dtype=tf.float32):
  x = tf.range(-2, 2, delta=0.1)
  X, Y = tf.meshgrid(x, x)
  z = tf.transpose(tf.reshape(tf.stack([X,Y], axis=0), [2,-1]))
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
                                        scale_diag=tf.ones([2], dtype=dtype))
  
  pi_z = proposal.prob(z)
  a_z = tf.squeeze(mlp(z, mlp_layers, name=mlp_name, final_activation=final_activation))

  unnorm_density = tf.reshape(pi_z*a_z, [40, 40])

  plot = plot_density(unnorm_density)
  tf.summary.image("density", plot, max_outputs=1, collections=["infrequent_summaries"])

# Code for NIS model
def make_nis_graph(batch_size=16,
                   K=100,
                   lr=1e-4,
                   mlp_layers=[10,10],
                   dtype=tf.float32):

  target_dist = mixture_of_nine()
  z_target = target_dist.sample(batch_size)

  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
                                        scale_diag=tf.ones([2], dtype=dtype))
  z_proposal = proposal.sample(K)

  neg_energy_target = tf.squeeze(mlp(z_target, mlp_layers, name="neg_energy", 
    final_activation=None))
  neg_energy_proposal = tf.squeeze(mlp(z_proposal, mlp_layers, name="neg_energy", 
    final_activation=None))  
 
  proposal_lse = tf.reduce_logsumexp(neg_energy_proposal, keepdims=True) 

  denom = (tf.reduce_logsumexp(tf.stack((neg_energy_target,
                      tf.tile(proposal_lse, [batch_size])), axis=-1), axis=-1) -
                      tf.log(tf.to_float(K+1)))

  lower_bound = (proposal.log_prob(z_target) + 
                 neg_energy_target -
                 tf.reduce_logsumexp(tf.stack((neg_energy_target,
                      tf.tile(proposal_lse, [batch_size])), axis=-1), axis=-1) + 
                 tf.log(tf.to_float(K+1)))
  lower_bound = tf.reduce_mean(lower_bound)

  tf.summary.scalar("lower_bound", lower_bound)
  #tf.summary.scalar("negative_energy_target", neg_energy_target)
  density_image_summary(mlp_name="neg_energy", mlp_layers=mlp_layers, final_activation=tf.math.exp)
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-lower_bound)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return lower_bound, train_op, global_step

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
        save_steps=10000,
        output_dir=FLAGS.logdir,
        summary_op=tf.summary.merge_all(key="infrequent_summaries")
    )
    hooks.append(infrequent_summary_hook)
  return hooks

def main(unused_argv):
  g = tf.Graph()
  with g.as_default():
    if FLAGS.algo == "lars":
      print("Running LARS")
      loss, train_op, global_step = make_lars_graph(
        batch_size=FLAGS.batch_size, 
	lr=FLAGS.learning_rate, 
	dtype=tf.float32)
    elif FLAGS.algo == "nis":
      print("Running NIS")
      loss, train_op, global_step = make_nis_graph(
        batch_size=FLAGS.batch_size,
        K=128,
        lr=FLAGS.learning_rate,
        mlp_layers=[20,20],
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
