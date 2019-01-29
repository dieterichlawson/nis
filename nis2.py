import os
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import functools
import tfmpl
import mnist_data
import base

tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_enum("algo", "nis_vae_proposal",
                         ["nis_vae_proposal", "nis_gaussian_proposal", 
                           "vae_gaussian_prior", "vae_nis_prior"],
                         "Algorithm to run.")
tf.app.flags.DEFINE_integer("latent_dim", 50,
                            "Dimension of the latent space of the VAE.")
tf.app.flags.DEFINE_integer("K", 128,
                            "Number of samples for NIS model.")
tf.app.flags.DEFINE_float("scale_min", 1e-5,
                             "Minimum scale for various distributions.")
tf.app.flags.DEFINE_float("learning_rate", 3e-4,
                           "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_string("logdir", "/tmp/nis",
                            "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS

def make_log_hooks(global_step, elbo):
  hooks = []
  def summ_formatter(d):
    return ("Step {step}, elbo: {elbo:.5f}".format(**d))
  elbo_hook = tf.train.LoggingTensorHook(
      {"step": global_step, "elbo": elbo},
      every_n_iter=FLAGS.summarize_every,
      formatter=summ_formatter)
  hooks.append(elbo_hook)
  if len(tf.get_collection("infrequent_summaries")) > 0:
    infrequent_summary_hook = tf.train.SummarySaverHook(
        save_steps=1000,
        output_dir=FLAGS.logdir,
        summary_op=tf.summary.merge_all(key="infrequent_summaries")
    )
    hooks.append(infrequent_summary_hook)
  return hooks

def sample_summary(model):
  ims = tf.reshape(model.sample(sample_shape=[FLAGS.batch_size]), [FLAGS.batch_size, 28, 28, 1])
  tf.summary.image("samples", ims, max_outputs=FLAGS.batch_size, 
                    collections=["infrequent_summaries"])

def main(unused_argv):
  FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.algo)
  g = tf.Graph()
  with g.as_default():

    data_batch, _, _ = mnist_data.get_mnist(
            batch_size=FLAGS.batch_size,
            split="train",
            binarized="dynamic")
    data_batch = tf.cast(data_batch, tf.float32)
    data_dim = data_batch.get_shape().as_list()[1]
    if FLAGS.algo == "nis_vae_proposal":
      print("Running NIS with VAE proposal")
      proposal = base.BernoulliVAE(
              latent_dim=FLAGS.latent_dim,
              data_dim=data_dim,
              decoder_hidden_sizes=[300, 300],
              q_hidden_sizes=[300, 300],
              scale_min=FLAGS.scale_min,
              dtype=tf.float32)
      nis = base.NIS(
              K=FLAGS.K,
              data_dim=data_dim,
              energy_hidden_sizes=[200, 100],
              proposal=proposal,
              dtype=tf.float32)
      elbo = nis.log_prob(data_batch)
    sample_summary(nis)
    # Finish constructing the graph
    elbo_avg = tf.reduce_mean(elbo)
    tf.summary.scalar("elbo", elbo_avg)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    grads = opt.compute_gradients(-elbo_avg)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    log_hooks = make_log_hooks(global_step, elbo_avg) 

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
