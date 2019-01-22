import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import functools
import tfmpl
import mnist_data

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


def reduce_logavgexp(input_tensor, axis=None, keepdims=None, name=None):
  dims = tf.shape(input_tensor)
  if axis is not None:
    dims = tf.gather(dims, axis)
  denominator = tf.reduce_prod(dims)
  return (tf.reduce_logsumexp(input_tensor, 
                              axis=axis, 
                              keepdims=keepdims, 
                              name=name) - tf.log(tf.to_float(denominator)))

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

def make_lars_loss(target_samples,
                   Z_batch_size=1024,
                   accept_fn_layers=[10, 10], 
                   log_Z_ema_decay=0.99,
                   dtype=tf.float32):
  # Create proposal as standard 2-D Gaussian
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=dtype),
                                        scale_diag=tf.ones([2], dtype=dtype))
 
  z_r = target_samples
  batch_size = tf.shape(target_samples)[0]
  
  # Compute log a(z), log pi(z), and log q(z)
  log_a_z_r = tf.reshape(mlp(z_r, accept_fn_layers, name="a"), [batch_size]) # [batch_size]
  log_pi_z_r = proposal.log_prob(z_r) # [batch_size]

  # Sample zs from proposal to estimate Z
  z_s = proposal.sample(Z_batch_size) # [Z_batch_size, 2]
  # Compute log a(z) for zs sampled from proposal
  log_a_z_s = tf.reshape(mlp(z_s, accept_fn_layers, name="a"), [Z_batch_size]) # [Z_batch_size]
  log_Z_curr = reduce_logavgexp(log_a_z_s) # []
  
  # Set up EMA of log_Z
  log_Z_ema = tf.train.ExponentialMovingAverage(decay=log_Z_ema_decay)
  log_Z_curr_sg = tf.stop_gradient(log_Z_curr)
  maintain_log_Z_ema_op = log_Z_ema.apply([log_Z_curr_sg])
  
  # In forward pass, log Z is the smoothed ema version of log Z
  # In backward pass it is the current estimate of log Z, log_Z_curr_avg
  log_Z = log_Z_curr + tf.stop_gradient(log_Z_ema.average(log_Z_curr_sg) - log_Z_curr)
  
  loss = -(log_pi_z_r + log_a_z_r - log_Z[tf.newaxis]) # [batch_size]

  tf.summary.scalar("log Z ema", log_Z_ema.average(log_Z_curr_sg))
  return tf.reduce_mean(loss), maintain_log_Z_ema_op

def make_lars_graph(target_samples,
                    lr=1e-4, 
                    mlp_layers=[10, 10], 
                    dtype=tf.float32):
  loss, ema_op = make_lars_loss(target_samples,
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
def make_nis_graph(target_samples,
                   K=100,
                   lr=1e-4,
                   mlp_layers=[10,10],
                   dtype=tf.float32):
  batch_size = tf.shape(target_samples)[0]
  data_dim = tf.shape(target_samples)[1]
  z_target = target_samples
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                        scale_diag=tf.ones([data_dim], dtype=dtype))
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
  #density_image_summary(mlp_name="neg_energy", mlp_layers=mlp_layers, final_activation=tf.math.exp)
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
    target_batch, _ = mnist_data.get_mnist(
            batch_size=FLAGS.batch_size,
            split="train")
    if FLAGS.algo == "lars":
      print("Running LARS")
      loss, train_op, global_step = make_lars_graph(
        target_batch,
        lr=FLAGS.learning_rate, 
        dtype=tf.float32)
      pass
    elif FLAGS.algo == "nis":
      print("Running NIS")
      loss, train_op, global_step = make_nis_graph(
        target_batch,
        K=128,
        lr=FLAGS.learning_rate,
        mlp_layers=[100, 50, 50, 20],
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
