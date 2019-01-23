import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import functools
import tfmpl
import mnist_data

tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_integer("latent_dim", 32,
                            "Dimension of the latent space of the VAE.")
tf.app.flags.DEFINE_integer("K", 128,
                            "Number of samples for NIS model.")
tf.app.flags.DEFINE_float("scale_min", 1e-5,
                             "Minimum scale for various distributions.")
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

def mlp(inputs, 
        layer_sizes,
        hidden_activation=tf.math.tanh,
        final_activation=tf.math.log_sigmoid,
        name=None):
  """Creates a simple multi-layer perceptron."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    for i, s in enumerate(layer_sizes[:-1]):
      inputs = tf.layers.dense(inputs,
                               units=s,
                               activation=hidden_activation,
                               kernel_initializer=tf.initializers.glorot_uniform,
                               name="layer_%d" % (i+1))
    output = tf.layers.dense(inputs,
                             units=layer_sizes[-1],
                             activation=final_activation,
                             kernel_initializer=tf.initializers.glorot_uniform,
                             name="layer_%d" % (len(layer_sizes)+1))
  return output

def conditional_normal(
        inputs,
        data_dim,
        hidden_sizes,
        hidden_activation=tf.math.tanh,
        scale_min=1e-5,
        name=None):
    raw_params = mlp(inputs, 
                     hidden_sizes + [2*data_dim],
                     hidden_activation=hidden_activation,
                     final_activation=None,
                     name=name)
    loc, raw_scale = tf.split(raw_params, 2, axis=-1)
    scale = tf.math.maximum(scale_min, tf.math.softplus(raw_scale))
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

def make_nis_lower_bound(target_samples, K, hidden_layer_sizes):
  """Constructs an NIS distribution for the given target samples and parameters.
  
  Args:
    target_samples: [batch_size, data_size]
  """
  batch_size = tf.shape(target_samples)[0]
  data_size = tf.shape(target_samples)[1]
  dtype = target_samples.dtype
  proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_size], dtype=dtype),
                                        scale_diag=tf.ones([data_size], dtype=dtype))
  proposal_samples = proposal.sample([batch_size, K])  #[batch_size, K, data_size]

  mlp_fn = functools.partial(
             mlp,
             layer_sizes=hidden_layer_sizes + [1],
             final_activation=None,
             name="nis_mlp")

  log_energy_target = tf.reshape(mlp_fn(target_samples), [batch_size])
  log_energy_proposal = tf.reshape(mlp_fn(proposal_samples), [batch_size, K])

  proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1) # [batch_size]
  
  #[batch_size]
  denom = tf.reduce_logsumexp(tf.stack([log_energy_target, proposal_lse], axis=-1), axis=1)
  denom -= tf.log(tf.to_float(K+1))

  lower_bound = proposal.log_prob(target_samples) +  log_energy_target - denom
  return lower_bound

def vae(data,
        latent_dim,
        q_fn,
        prior_fn,
        generative_fn):
  batch_size = tf.shape(data)[0]
  data_dim = data.get_shape().as_list()[1]

  # Construct approximate posterior and sample z.
  q = q_fn(data)
  z = q.sample()
  log_q_z = q.log_prob(z)

  # compute the prior prob of z
  log_p_z = prior_fn(z)

  # Compute the model logprob of the data 
  p_x_given_z = generative_fn(z)
  log_p_x_given_z = p_x_given_z.log_prob(data)

  elbo = log_p_z + log_p_x_given_z - log_q_z
  return elbo

def make_vae_with_nis_prior(
        data,
        latent_dim,
        K,
        nis_hidden_sizes,
        q_hidden_sizes,
        p_x_hidden_sizes,
        scale_min=1e-5,
        lr=1e-4):

  q_fn = functools.partial(
          conditional_normal,
          data_dim=latent_dim,
          hidden_sizes=q_hidden_sizes,
          scale_min=scale_min,
          name="q")

  prior_fn = functools.partial(
          make_nis_lower_bound, 
          K=K, 
          hidden_layer_sizes=nis_hidden_sizes)

  generative_fn = functools.partial(
          conditional_normal,
          data_dim=data.get_shape().as_list()[1],
          hidden_sizes=p_x_hidden_sizes,
          scale_min=scale_min,
          name="generative")

  elbo = vae(data,
             latent_dim,
             q_fn,
             prior_fn,
             generative_fn)
  
  elbo_avg = tf.reduce_mean(elbo)
  tf.summary.scalar("elbo", elbo_avg)
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-elbo_avg)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return elbo_avg, train_op, global_step

def make_nis_with_vae_proposal(
        data,
        K,
        vae_latent_dim,
        nis_hidden_sizes,
        q_hidden_sizes,
        p_x_hidden_sizes,
        scale_min=1e-5,
        lr=1e-4):
  batch_size = data.get_shape().as_list()[0]
  data_size = data.get_shape().as_list()[1]
  dtype = data.dtype
  # Sample z_1:K from the VAE
  vae_prior = tfd.MultivariateNormalDiag(loc=tf.zeros([vae_latent_dim], dtype=dtype),
                                        scale_diag=tf.ones([vae_latent_dim], dtype=dtype))
  z = vae_prior.sample([batch_size, K])  #[batch_size, K, vae_latent_dim]
  # Use zs to sample xs
  p_x_given_z_fn = functools.partial(conditional_normal,
          data_dim=data_size, 
          hidden_sizes=p_x_hidden_sizes, 
          scale_min=scale_min, 
          name="p_x_given_z")
  p_x_given_z = p_x_given_z_fn(z)
  x = p_x_given_z.sample() #[batch_size, K, data_size]

  # compute lower bound on log prob of data
  q = conditional_normal(
          data, vae_latent_dim, q_hidden_sizes, scale_min=scale_min, name="q")
  z_q = q.sample() # [batch_size, vae_latent_dim]
  p_x_given_z_q = p_x_given_z_fn(z_q)
  log_p_data_lb = vae_prior.log_prob(z_q) + p_x_given_z_q.log_prob(data) - q.log_prob(z_q)
  
  mlp_fn = functools.partial(
             mlp,
             layer_sizes=nis_hidden_sizes+ [1],
             final_activation=None,
             name="nis_mlp")

  log_energy_target = tf.reshape(mlp_fn(data), [batch_size])
  log_energy_proposal = tf.reshape(mlp_fn(x), [batch_size, K])

  proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1) # [batch_size]
  
  #[batch_size]
  denom = tf.reduce_logsumexp(tf.stack([log_energy_target, proposal_lse], axis=-1), axis=1)
  denom -= tf.log(tf.to_float(K+1))

  lower_bound = log_p_data_lb + log_energy_target - denom
  return lower_bound

def make_nis_with_vae_proposal_graph(
        data,
        K,
        vae_latent_dim,
        nis_hidden_sizes,
        q_hidden_sizes,
        p_x_hidden_sizes,
        scale_min=1e-5,
        lr=1e-4):
  elbo = make_nis_with_vae_proposal(data, K, vae_latent_dim, nis_hidden_sizes, 
          q_hidden_sizes, p_x_hidden_sizes,
          scale_min=scale_min, lr=lr)
  elbo_avg = tf.reduce_mean(elbo)
  tf.summary.scalar("elbo", elbo_avg)
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-elbo_avg)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return elbo_avg, train_op, global_step

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
    print("Running VAE with NIS prior")

    data_batch, _, _ = mnist_data.get_mnist(
            batch_size=FLAGS.batch_size,
            split="train")

    #loss, train_op, global_step = make_vae_with_nis_prior(
    loss, train_op, global_step = make_nis_with_vae_proposal_graph(
      data_batch,
      K=FLAGS.K,
      vae_latent_dim=FLAGS.latent_dim,
      nis_hidden_sizes=[200,100],
      q_hidden_sizes=[100,50],
      p_x_hidden_sizes=[100,250],
      scale_min=FLAGS.scale_min,
      lr=FLAGS.learning_rate)
        
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
      while cur_step <= FLAGS.max_steps and not sess.should_stop():
        _, cur_step = sess.run([train_op, global_step])

if __name__ == "__main__":
  tf.app.run(main)
