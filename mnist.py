import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import functools
import tfmpl
import datasets
import base

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_enum("mode", "train", ["train", "eval"],
                         "Mode to run.")
tf.app.flags.DEFINE_enum("dataset", "raw_mnist",
                         ["raw_mnist", "dynamic_mnist", "static_mnist", "nine_gaussians"],
                         "Dataset to use.")
tf.app.flags.DEFINE_enum("proposal", "bernoulli_vae",
                        ["bernoulli_vae","gaussian_vae","gaussian"],
                        "Proposal type to use.")
tf.app.flags.DEFINE_enum("model", "bernoulli_vae",
                        ["bernoulli_vae","gaussian_vae","nis"],
                        "Model type to use.")

tf.app.flags.DEFINE_integer("latent_dim", 50,
                            "Dimension of the latent space of the VAE.")
tf.app.flags.DEFINE_integer("K", 128,
                            "Number of samples for NIS model.")
tf.app.flags.DEFINE_float("scale_min", 1e-5,
                             "Minimum scale for various distributions.")
tf.app.flags.DEFINE_float("learning_rate", 3e-4,
                           "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_boolean("decay_lr", True,
                            "Divide the learning rate by 3 every 1e6 iterations.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_string("split", "train",
                           "The dataset split to train on.")
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

def get_dataset(dataset, batch_size, split, repeat=True, shuffle=True, initializable=False):
  if dataset == "dynamic_mnist":
    data_batch, mean, itr = datasets.get_dynamic_mnist(batch_size=batch_size, split=split, 
            repeat=repeat, shuffle=shuffle, initializable=initializable)
  elif dataset == "raw_mnist":
    data_batch, mean, itr = datasets.get_raw_mnist(batch_size=batch_size, split=split,
            repeat=repeat, shuffle=shuffle, initializable=initializable)
  elif dataset == "static_mnist":
    data_batch, mean, itr = datasets.get_static_mnist(batch_size=batch_size, split=split,
            repeat=repeat, shuffle=shuffle, initializable=initializable)

  return tf.cast(data_batch, tf.float32), mean, itr

def make_model(proposal_type, model_type, data_dim, mean):
  if proposal_type == "bernoulli_vae":
    proposal = base.BernoulliVAE(
            latent_dim=FLAGS.latent_dim,
            data_dim=data_dim,
            decoder_hidden_sizes=[300, 300],
            q_hidden_sizes=[300, 300],
            scale_min=FLAGS.scale_min,
            dtype=tf.float32)
  elif proposal_type == "gaussian_vae":
    proposal = base.GaussianVAE(
            latent_dim=FLAGS.latent_dim,
            data_dim=data_dim,
            decoder_hidden_sizes=[300, 300],
            q_hidden_sizes=[300, 300],
            scale_min=FLAGS.scale_min,
            truncate=False,
            dtype=tf.float32)
  elif proposal_type == "gaussian":
    proposal = tfd.MultivariateNormalDiag(
            loc=tf.zeros([FLAGS.latent_dim], dtype=tf.float32),
            scale_diag=tf.ones([FLAGS.latent_dim], dtype=tf.float32))

  if model_type == "bernoulli_vae":
    model = base.BernoulliVAE(
            latent_dim=FLAGS.latent_dim,
            data_dim=data_dim,
            decoder_hidden_sizes=[300, 300],
            q_hidden_sizes=[300, 300],
            scale_min=FLAGS.scale_min,
            bias_init=mean,
            prior=proposal,
            dtype=tf.float32)
  elif model_type == "gaussian_vae":
    model = base.GaussianVAE(
            latent_dim=FLAGS.latent_dim,
            data_dim=data_dim,
            decoder_hidden_sizes=[300, 300],
            q_hidden_sizes=[300, 300],
            scale_min=FLAGS.scale_min,
            truncate=False,
            prior=proposal,
            dtype=tf.float32)
  elif model_type == "nis":
    model = base.NIS(
            K=FLAGS.K,
            data_dim=data_dim,
            energy_hidden_sizes=[20, 20],
            proposal=proposal,
            dtype=tf.float32)
  return model

def run_train():
  dirname = "_".join([FLAGS.dataset, FLAGS.proposal, "proposal", FLAGS.model, "model"])
  FLAGS.logdir = os.path.join(FLAGS.logdir, dirname)
  g = tf.Graph()
  with g.as_default():
    data_batch, mean, _ = get_dataset(
            FLAGS.dataset, 
            batch_size=FLAGS.batch_size,
            split=FLAGS.split,
            repeat=True, 
            shuffle=True)
    data_dim = data_batch.get_shape().as_list()[1]
    model = make_model(FLAGS.proposal, FLAGS.model, data_dim, mean)
    elbo = model.log_prob(data_batch)
    sample_summary(model)
    # Finish constructing the graph
    elbo_avg = tf.reduce_mean(elbo)
    tf.summary.scalar("elbo", elbo_avg)
    global_step = tf.train.get_or_create_global_step()
    if FLAGS.decay_lr:
      lr = tf.train.exponential_decay(
              FLAGS.learning_rate,
              global_step,
              decay_steps=int(1e6),
              decay_rate=1./3.,
              staircase=True)
    else:
      lr = FLAGS.learning_rate
    tf.summary.scalar("learning_rate", lr)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(-elbo_avg)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    log_hooks = make_log_hooks(global_step, elbo_avg) 

    with tf.train.MonitoredTrainingSession(
        master="",
        is_chief=True,
        hooks=log_hooks,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=60,
        save_summaries_steps=FLAGS.summarize_every,
        log_step_count_steps=FLAGS.summarize_every) as sess:
      cur_step = -1
      while cur_step <= FLAGS.max_steps and not sess.should_stop():
        _, cur_step = sess.run([train_op, global_step])

def average_elbo_over_dataset(bound, batch_size, sess):
  total_ll = np.zeros(1, dtype=np.float64)
  total_n_elems = 0.0
  while True:
    try:
      outs = sess.run([bound, batch_size])
    except tf.errors.OutOfRangeError:
      break
    total_ll += outs[0]
    total_n_elems += outs[1]
  ll_per_elem = total_ll / total_n_elems
  return ll_per_elem

def restore_checkpoint_if_exists(saver, sess, logdir):
  """Looks for a checkpoint and restores the session from it if found.
  Args:
    saver: A tf.train.Saver for restoring the session.
    sess: A TensorFlow session.
    logdir: The directory to look for checkpoints in.
  Returns:
    True if a checkpoint was found and restored, False otherwise.
  """
  checkpoint = tf.train.get_checkpoint_state(logdir)
  if checkpoint:
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    full_checkpoint_path = os.path.join(logdir, checkpoint_name)
    saver.restore(sess, full_checkpoint_path)
    return True
  return False

def wait_for_checkpoint(saver, sess, logdir):
  """Loops until the session is restored from a checkpoint in logdir.
  Args:
    saver: A tf.train.Saver for restoring the session.
    sess: A TensorFlow session.
    logdir: The directory to look for checkpoints in.
  """
  while not restore_checkpoint_if_exists(saver, sess, logdir):
    tf.logging.info("Checkpoint not found in %s, sleeping for 60 seconds."
                    % logdir)
    time.sleep(60)

def run_eval():
  g = tf.Graph()
  with g.as_default():
    global_step = tf.train.get_or_create_global_step()
    data_batch, mean, itr = get_dataset(
            FLAGS.dataset, 
            batch_size=FLAGS.batch_size,
            split=FLAGS.split,
            repeat=False, 
            shuffle=False,
            initializable=True)
    batch_size = tf.shape(data_batch)[0]
    data_dim = data_batch.get_shape().as_list()[1]
    model = make_model(FLAGS.proposal, FLAGS.model, data_dim, mean)
    elbo = model.log_prob(data_batch)
    elbo_sum = tf.reduce_sum(elbo)
    summary_dir = os.path.join(FLAGS.logdir, FLAGS.split)
    summary_writer = tf.summary.FileWriter(
        summary_dir, flush_secs=15, max_queue=100)
    saver = tf.train.Saver()
    prev_evaluated_step = -1
    with tf.train.SingularMonitoredSession() as sess:
      while True:
        wait_for_checkpoint(saver, sess, FLAGS.logdir)
        step = sess.run(global_step)
        tf.logging.info("Model restored from step %d." % step)
        if step == prev_evaluated_step:
          tf.logging.info("Already evaluated checkpoint at step %d, sleeping" % step)
          time.sleep(60)
          continue
        sess.run(itr.initializer)
        average_elbo = average_elbo_over_dataset(elbo_sum, batch_size, sess)
        prev_evaluated_step = step

        value = tf.Summary.Value(tag="%s_elbo" % FLAGS.split, simple_value=average_elbo[0])
        summary = tf.Summary(value=[value])
        summary_writer.add_summary(summary, global_step=step)
        print("elbo: %f" % average_elbo[0])
    
def main(unused_argv):
  if FLAGS.mode == "train":
    run_train()
  else:
    run_eval()
  
if __name__ == "__main__":
  tf.app.run(main)