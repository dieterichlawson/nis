from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import time
import google3
import tensorflow as tf
import tensorflow_probability as tfp
import google3.experimental.users.gjt.his.datasets as datasets
from google3.experimental.users.gjt.his.models import his
from google3.experimental.users.gjt.his.models import maf
from google3.experimental.users.gjt.his.models import nis
from google3.experimental.users.gjt.his.models import vae
import google3.learning.deepmind.xmanager2.client.google as xm

tfd = tfp.distributions


flags.DEFINE_enum("mode", "train", ["train", "eval"], "Mode to run.")
flags.DEFINE_enum(
    "dataset", "raw_mnist",
    ["raw_mnist", "jittered_mnist", "dynamic_mnist", "static_mnist",
     "jittered_celeba"],
    "Dataset to use.")
flags.DEFINE_enum(
    "proposal", "bernoulli_vae",
    ["bernoulli_vae", "gaussian_vae", "gaussian", "nis", "bnis", "maf"],
    "Proposal type to use.")
flags.DEFINE_enum(
    "model", "bernoulli_vae",
    ["bernoulli_vae", "gaussian_vae", "nis", "bnis", "his", "maf", "hisvae",
     "conv_gaussian_vae"],
    "Model type to use.")
flags.DEFINE_string(
    "decoder_hidden_sizes", "300,300",
    "Comma-delimited list denoting the hidden sizes of the VAE decoder.")
flags.DEFINE_string("q_hidden_sizes", "300,300",
                    "Comma-delimited list denoting the hidden sizes of q.")
flags.DEFINE_string(
    "energy_hidden_sizes", "100,100",
    "Comma-delimited list denoting the hidden sizes of the energy function.")
flags.DEFINE_boolean("reparameterize_proposal", True,
                     "If true, reparameterize the samples of the prior.")
flags.DEFINE_boolean(
    "squash", False,
    "If true, squash the output of the normal prior to be between 0 and 1.")
flags.DEFINE_float(
    "gst_temperature", 0.7,
    "Default temperature for the Gumbel straight-through relaxation.")
flags.DEFINE_boolean(
    "vae_decoder_nn_scale", False,
    "If true, the scale of the data in the VAE is defined by a NN."
    "If false, it is defined by a learnable per-dimension constant.")
flags.DEFINE_boolean("learn_his_temps", True,
                     "If true, the annealing schedule of HIS is learnable.")
flags.DEFINE_boolean("learn_his_stepsize", True,
                     "If true, the step size of HIS is learnable.")
flags.DEFINE_float("his_init_alpha", 0.9995, "Initial alpha for HIS.")
flags.DEFINE_float("his_init_stepsize", 0.1, "Initial stepsize for HIS.")
flags.DEFINE_integer(
    "his_T", 5,
    "The number of timesteps to run Hamiltonian dynamics for in HIS.")
flags.DEFINE_integer("latent_dim", 50,
                     "Dimension of the latent space of the VAE.")
flags.DEFINE_integer("base_depth", 32,
                     "Base depth for convs.")
flags.DEFINE_integer("K", 128, "Number of samples for NIS model.")
flags.DEFINE_string(
    "num_iwae_samples", "1",
    "Number of samples used for IWAE bound during eval."
    "Can be a comma-separated list of integers.")
flags.DEFINE_float("scale_min", 1e-5,
                   "Minimum scale for various distributions.")
flags.DEFINE_float("learning_rate", 3e-4,
                   "The learning rate to use for ADAM or SGD.")
flags.DEFINE_integer(
    "anneal_kl_step", -1,
    "Anneal kl weight from 0 to 1 linearly, ending on this step."
    "If running eval, annealing is not used.")
flags.DEFINE_boolean("decay_lr", True,
                     "Divide the learning rate by 3 every 1e6 iterations.")
flags.DEFINE_integer("batch_size", 128, "The number of examples per batch.")
flags.DEFINE_string("split", "train", "The dataset split to train on.")
flags.DEFINE_string("logdir", None,
                    "Directory for summaries and checkpoints.")
flags.DEFINE_integer("max_steps", int(1e7),
                     "The number of steps to run training for.")
flags.DEFINE_integer("summarize_every", int(1e4),
                     "The number of steps between each evaluation.")
flags.DEFINE_integer(
    "num_summary_ims", 8,
    "The number of images to sample from the model for evaluation.")
flags.DEFINE_integer("flow_layers", 5,
                     "The number of flow layers to use in the MAF model.")
flags.DEFINE_integer(
    "run", 0,
    ("A number to distinguish which run this is. This allows us to run ",
     "multiple trials with the same params."))

FLAGS = flags.FLAGS


def exp_name():
  return "proposal-%s.model-%s.his_T-%d.run-%d" % (
      FLAGS.proposal,
      FLAGS.model,
      FLAGS.his_T,
      FLAGS.run,)


def print_flags():
  tf.logging.info("Running mnist.py with arguments.")
  for k, v in flags.FLAGS.flag_values_dict().items():
    tf.logging.info("    %s: %s", k, v)


def make_log_hooks(global_step, elbo):
  hooks = []

  def summ_formatter(d):
    return ("Step {step}, elbo: {elbo:.5f}".format(**d))

  elbo_hook = tf.train.LoggingTensorHook({
      "step": global_step,
      "elbo": elbo
  },
                                         every_n_iter=FLAGS.summarize_every,
                                         formatter=summ_formatter)
  hooks.append(elbo_hook)
  if len(tf.get_collection("infrequent_summaries")) > 0:
    infrequent_summary_hook = tf.train.SummarySaverHook(
        save_steps=FLAGS.summarize_every*10,
        output_dir=os.path.join(FLAGS.logdir, exp_name()),
        summary_op=tf.summary.merge_all(key="infrequent_summaries"))
    hooks.append(infrequent_summary_hook)
  return hooks


def sample_summary(model, data_dim):
  if data_dim == 784:
    data_dim = [28, 28, 1]
  ims = tf.reshape(
      model.sample(sample_shape=[FLAGS.num_summary_ims]),
      [FLAGS.num_summary_ims] + data_dim)
  tf.summary.image(
      "samples",
      ims,
      max_outputs=FLAGS.num_summary_ims,
      collections=["infrequent_summaries"])


def get_dataset(dataset,
                batch_size,
                split,
                repeat=True,
                shuffle=True,
                initializable=False):
  if dataset == "dynamic_mnist":
    data_batch, mean, itr = datasets.get_dynamic_mnist(
        batch_size=batch_size,
        split=split,
        repeat=repeat,
        shuffle=shuffle,
        initializable=initializable)
  elif dataset == "raw_mnist":
    data_batch, mean, itr = datasets.get_raw_mnist(
        batch_size=batch_size,
        split=split,
        repeat=repeat,
        shuffle=shuffle,
        initializable=initializable)
  elif dataset == "static_mnist":
    data_batch, mean, itr = datasets.get_static_mnist(
        batch_size=batch_size,
        split=split,
        repeat=repeat,
        shuffle=shuffle,
        initializable=initializable)
  elif dataset == "jittered_mnist":
    data_batch, mean, itr = datasets.get_jittered_mnist(
        batch_size=batch_size,
        split=split,
        repeat=repeat,
        shuffle=shuffle,
        initializable=initializable)
  elif dataset == "jittered_celeba":
    data_batch, mean, itr = datasets.get_jittered_celeba(
        batch_size=batch_size,
        split=split)
  return tf.cast(data_batch, tf.float32), mean, itr


def make_kl_weight(global_step, anneal_kl_step):
  if anneal_kl_step > 0:
    kl_weight = tf.clip_by_value(
        tf.to_float(global_step) / tf.to_float(anneal_kl_step), 0., 1.)
  else:
    kl_weight = 1.
  tf.summary.scalar("kl_weight", kl_weight)
  return kl_weight


def make_model(proposal_type, model_type, data_dim, mean, global_step):
  kl_weight = make_kl_weight(global_step, FLAGS.anneal_kl_step)
  # Bernoulli VAE proposal gets that data mean because it is proposing images.
  # Other proposals don't because they are proposing latent states.
  decoder_hidden_sizes = [
      int(x.strip()) for x in FLAGS.decoder_hidden_sizes.split(",")
  ]
  q_hidden_sizes = [int(x.strip()) for x in FLAGS.q_hidden_sizes.split(",")]
  energy_hidden_sizes = [
      int(x.strip()) for x in FLAGS.energy_hidden_sizes.split(",")
  ]
  if model_type in ["nis", "bnis", "maf", "his"]:
    proposal_data_dim = data_dim
  elif model_type in ["bernoulli_vae", "gaussian_vae", "hisvae",
                      "conv_gaussian_vae"]:
    proposal_data_dim = FLAGS.latent_dim

  if proposal_type == "bernoulli_vae":
    proposal = vae.BernoulliVAE(
        latent_dim=FLAGS.latent_dim,
        data_dim=proposal_data_dim,
        data_mean=mean,
        decoder_hidden_sizes=decoder_hidden_sizes,
        q_hidden_sizes=q_hidden_sizes,
        scale_min=FLAGS.scale_min,
        kl_weight=kl_weight,
        reparameterize_sample=FLAGS.reparameterize_proposal,
        temperature=FLAGS.gst_temperature,
        dtype=tf.float32)
  elif proposal_type == "gaussian_vae":
    proposal = vae.GaussianVAE(
        latent_dim=FLAGS.latent_dim,
        data_dim=proposal_data_dim,
        data_mean=mean,
        decoder_hidden_sizes=decoder_hidden_sizes,
        decoder_nn_scale=FLAGS.vae_decoder_nn_scale,
        q_hidden_sizes=q_hidden_sizes,
        scale_min=FLAGS.scale_min,
        kl_weight=kl_weight,
        squash=FLAGS.squash,
        dtype=tf.float32)
  elif proposal_type == "nis":
    proposal = nis.NIS(
        K=FLAGS.K,
        data_dim=proposal_data_dim,
        energy_hidden_sizes=energy_hidden_sizes,
        dtype=tf.float32)
  elif proposal_type == "gaussian":
    proposal = tfd.MultivariateNormalDiag(
        loc=tf.zeros([proposal_data_dim], dtype=tf.float32),
        scale_diag=tf.ones([proposal_data_dim], dtype=tf.float32))
  elif proposal_type == "bnis":
    proposal = nis.BernoulliNIS(
        K=FLAGS.K,
        data_dim=proposal_data_dim,
        energy_hidden_sizes=energy_hidden_sizes,
        q_hidden_sizes=q_hidden_sizes,
        temperature=FLAGS.gst_temperature,
        dtype=tf.float32)
  elif proposal_type == "maf":
    proposal = maf.MAF(
        data_dim=proposal_data_dim,
        hidden_sizes=energy_hidden_sizes,
        flow_layers=FLAGS.flow_layers,
        dtype=tf.float32)

  if model_type == "bernoulli_vae":
    model = vae.BernoulliVAE(
        latent_dim=FLAGS.latent_dim,
        data_dim=data_dim,
        data_mean=mean,
        decoder_hidden_sizes=decoder_hidden_sizes,
        q_hidden_sizes=q_hidden_sizes,
        scale_min=FLAGS.scale_min,
        prior=proposal,
        kl_weight=kl_weight,
        reparameterize_sample=False,
        dtype=tf.float32)
  elif model_type == "gaussian_vae":
    model = vae.GaussianVAE(
        latent_dim=FLAGS.latent_dim,
        data_dim=data_dim,
        data_mean=mean,
        decoder_hidden_sizes=decoder_hidden_sizes,
        decoder_nn_scale=FLAGS.vae_decoder_nn_scale,
        q_hidden_sizes=q_hidden_sizes,
        scale_min=FLAGS.scale_min,
        prior=proposal,
        kl_weight=kl_weight,
        dtype=tf.float32,
        squash=FLAGS.squash)
  elif model_type == "conv_gaussian_vae":
    model = vae.ConvGaussianVAE(
        latent_dim=FLAGS.latent_dim,
        data_dim=data_dim,
        data_mean=mean,
        base_depth=FLAGS.base_depth,
        scale_min=FLAGS.scale_min,
        prior=proposal,
        kl_weight=kl_weight,
        dtype=tf.float32,
        squash=FLAGS.squash)
  elif model_type == "nis":
    model = nis.NIS(
        K=FLAGS.K,
        data_dim=data_dim,
        data_mean=mean,
        energy_hidden_sizes=energy_hidden_sizes,
        proposal=proposal,
        reparameterize_proposal_samples=FLAGS.reparameterize_proposal,
        dtype=tf.float32)
  elif model_type == "bnis":
    model = nis.BernoulliNIS(
        K=FLAGS.K,
        data_dim=data_dim,
        data_mean=mean,
        energy_hidden_sizes=energy_hidden_sizes,
        q_hidden_sizes=q_hidden_sizes,
        proposal=proposal,
        temperature=FLAGS.gst_temperature,
        dtype=tf.float32)
  elif model_type == "his":
    model = his.HIS(
        proposal=proposal,
        T=FLAGS.his_T,
        data_dim=data_dim,
        data_mean=mean,
        energy_hidden_sizes=energy_hidden_sizes,
        q_hidden_sizes=q_hidden_sizes,
        learn_temps=FLAGS.learn_his_temps,
        learn_stepsize=FLAGS.learn_his_stepsize,
        init_alpha=FLAGS.his_init_alpha,
        init_step_size=FLAGS.his_init_stepsize,
        squash=FLAGS.squash,
        dtype=tf.float32)
  elif model_type == "hisvae":
    model = his.HISVAE(
        latent_dim=FLAGS.latent_dim,
        proposal=proposal,
        T=FLAGS.his_T,
        data_dim=data_dim,
        data_mean=mean,
        energy_hidden_sizes=energy_hidden_sizes,
        q_hidden_sizes=q_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        learn_temps=FLAGS.learn_his_temps,
        learn_stepsize=FLAGS.learn_his_stepsize,
        init_alpha=FLAGS.his_init_alpha,
        init_step_size=FLAGS.his_init_stepsize,
        squash=FLAGS.squash,
        kl_weight=kl_weight,
        dtype=tf.float32)
  elif model_type == "maf":
    model = maf.MAF(
        data_dim=data_dim,
        hidden_sizes=energy_hidden_sizes,
        flow_layers=FLAGS.flow_layers,
        proposal=proposal,
        dtype=tf.float32)

  return model


def run_train():
  print_flags()
  g = tf.Graph()
  with g.as_default():
    global_step = tf.train.get_or_create_global_step()
    data_batch, mean, _ = get_dataset(
        FLAGS.dataset,
        batch_size=FLAGS.batch_size,
        split=FLAGS.split,
        repeat=True,
        shuffle=True)
    data_dim = data_batch.get_shape().as_list()[1:]
    if len(data_dim) == 1:
      data_dim = data_dim[0]
    model = make_model(FLAGS.proposal, FLAGS.model, data_dim, mean, global_step)
    elbo = model.log_prob(data_batch)
    sample_summary(model, data_dim)
    # Finish constructing the graph
    elbo_avg = tf.reduce_mean(elbo)
    tf.summary.scalar("elbo", elbo_avg)
    if FLAGS.decay_lr:
      #lr = tf.train.exponential_decay(
      #        FLAGS.learning_rate,
      #        global_step,
      #        decay_steps=int(1e6),
      #        decay_rate=1./3.,
      #        staircase=True)
      lr = tf.train.piecewise_constant(
          global_step, [int(1e6)],
          [FLAGS.learning_rate, FLAGS.learning_rate / 3.])
    else:
      lr = FLAGS.learning_rate
    tf.summary.scalar("learning_rate", lr)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(-elbo_avg)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    log_hooks = make_log_hooks(global_step, elbo_avg)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.train.MonitoredTrainingSession(
        master="",
        config=config,
        is_chief=True,
        hooks=log_hooks,
        checkpoint_dir=os.path.join(FLAGS.logdir, exp_name()),
        save_checkpoint_steps=FLAGS.summarize_every * 2,
        save_summaries_steps=FLAGS.summarize_every,
        log_step_count_steps=FLAGS.summarize_every) as sess:
      cur_step = -1
      while cur_step <= FLAGS.max_steps and not sess.should_stop():
        _, cur_step = sess.run([train_op, global_step])


def average_elbo_over_dataset(bound, batch_size, sess):
  total_ll = 0.0
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
    tf.logging.info("Checkpoint not found in %s, sleeping for 60 seconds." %
                    logdir)
    time.sleep(30)


def run_eval():
  print_flags()
  g = tf.Graph()
  with g.as_default():
    # If running eval, do not anneal the KL.
    FLAGS.anneal_kl_step = -1
    global_step = tf.train.get_or_create_global_step()
    summary_dir = os.path.join(FLAGS.logdir, exp_name(), "eval")
    summary_writer = tf.summary.FileWriter(
        summary_dir, flush_secs=15, max_queue=100)

    splits = FLAGS.split.split(",")
    for split in splits:
      assert split in ["train", "test", "valid"]

    num_iwae_samples = [
        int(x.strip()) for x in FLAGS.num_iwae_samples.split(",")
    ]
    assert len(num_iwae_samples) == 1 or len(num_iwae_samples) == len(splits)
    if len(num_iwae_samples) == 1:
      num_iwae_samples = num_iwae_samples * len(splits)

    bound_names = []
    for ns in num_iwae_samples:
      if ns > 1:
        bound_names.append("iwae_%d" % ns)
      else:
        bound_names.append("elbo")

    itrs = []
    batch_sizes = []
    elbos = []
    model = None
    for split, num_samples in zip(splits, num_iwae_samples):
      data_batch, mean, itr = get_dataset(
          FLAGS.dataset,
          batch_size=FLAGS.batch_size,
          split=split,
          repeat=False,
          shuffle=False,
          initializable=True)
      itrs.append(itr)
      batch_sizes.append(tf.shape(data_batch)[0])
      if model is None:
        data_dim = data_batch.get_shape().as_list()[1:]
        if len(data_dim) == 1:
          data_dim = data_dim[0]
        model = make_model(FLAGS.proposal, FLAGS.model, data_dim, mean,
                           global_step)
      elbos.append(
          tf.reduce_sum(model.log_prob(data_batch, num_samples=num_samples)))

    saver = tf.train.Saver()
    prev_evaluated_step = -1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.train.SingularMonitoredSession(config=config) as sess:
      while True:
        wait_for_checkpoint(saver, sess, os.path.join(
            FLAGS.logdir, exp_name()))
        step = sess.run(global_step)
        tf.logging.info("Model restored from step %d." % step)
        if step == prev_evaluated_step:
          tf.logging.info("Already evaluated checkpoint at step %d, sleeping" %
                          step)
          time.sleep(30)
          continue
        for i in range(len(splits)):
          sess.run(itrs[i].initializer)
          avg_elbo = average_elbo_over_dataset(elbos[i], batch_sizes[i], sess)
          value = tf.Summary.Value(
              tag="%s_%s" % (splits[i], bound_names[i]), simple_value=avg_elbo)
          summary = tf.Summary(value=[value])
          summary_writer.add_summary(summary, global_step=step)
          tf.logging.info("Step %d, %s %s: %f" %
                          (step, splits[i], bound_names[i], avg_elbo))
        prev_evaluated_step = step


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.mode == "train":
    run_train()
  else:
    run_eval()


if __name__ == "__main__":
  app.run(main)
