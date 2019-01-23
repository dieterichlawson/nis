import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools

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

def conditional_bernoulli(
        inputs,
        data_dim,
        hidden_sizes,
        hidden_activation=tf.math.tanh,
        name=None):
    bern_logits = mlp(inputs,
                      hidden_sizes + [data_dim],
                      hidden_activation=hidden_activation,
                      final_activation=None,
                      name=name)
    return tfd.Bernoulli(logits=bern_logits)


class VAE(object):
  """Variational autoencoder with continuous latent space."""

  def __init__(self, 
               latent_dim,
               decoder,
               q_hidden_sizes,
               prior=None,
               scale_min=1e-5,
               dtype=tf.float32,
               name="vae"):
    """Creates a VAE.

    Args:
      latent_dim: The size of the latent variable of the VAE.
      decoder: A callable that accepts a batch of latent samples and returns a distribution
        over the data space of the VAE. The distribution should support sample() and
        log_prob().
      q_hidden_sizes: A list of python ints, the sizes of the hidden layers in the MLP
        that parameterizes the q of this VAE.
      prior: A distribution over the clatent space of the VAE. The object must support 
        sample() and log_prob(). If not provided, defaults to Gaussian.
    """
    self.decoder = decoder
    with tf.name_scope(name):
      self.q = functools.partial(
            conditional_normal,
            data_dim=latent_dim,
            hidden_sizes=q_hidden_sizes,
            scale_min=scale_min,
            name="q")

    if prior is None:
      self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros([latent_dim], dtype=dtype),
                                              scale_diag=tf.ones([latent_dim], dtype=dtype))
    else:
      self.prior = prior

  def log_prob(self, data):
    batch_size = tf.shape(data)[0]
    data_dim = data.get_shape().as_list()[1]

    # Construct approximate posterior and sample z.
    q_z = self.q(data)
    z = q_z.sample()
    log_q_z = q_z.log_prob(z)

    # compute the prior prob of z
    log_p_z = self.prior.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    log_p_x_given_z = p_x_given_z.log_prob(data)

    elbo = log_p_z + log_p_x_given_z - log_q_z
    return elbo

  def sample(self, sample_shape=[1]):
    z = self.prior.sample(sample_shape)
    p_x_given_z = self.decoder(z)
    return p_x_given_z.sample()

class GaussianVAE(VAE):
  """VAE with Gaussian generative distribution."""
 
  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               prior=None,
               scale_min=1e-5,
               dtype=tf.float32,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    with tf.name_scope(name):
      decoder_fn = functools.partial(
            conditional_normal,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            scale_min=scale_min,
            name="decoder")

    super().__init__(
            latent_dim=latent_dim, 
            decoder=decoder_fn, 
            q_hidden_sizes=q_hidden_sizes, 
            prior=prior, 
            scale_min=scale_min, 
            dtype=dtype, 
            name=name)

class BernoulliVAE(VAE):
  """VAE with Gaussian generative distribution."""
 
  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               prior=None,
               scale_min=1e-5,
               dtype=tf.float32,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    with tf.name_scope(name):
      decoder_fn = functools.partial(
            conditional_bernoulli,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            name="decoder")

    super().__init__(latent_dim, decoder_fn, q_hidden_sizes, 
            prior=prior, scale_min=scale_min, dtype=dtype, name=name)

class NIS(object):

  def __init__(self,
               K, 
               data_dim, 
               energy_hidden_sizes, 
               proposal=None,
               dtype=tf.float32,
               name="nis"):
    """Creates a NIS model.
    
    Args:
      K: The number of proposal samples to take.
      data_dim: The dimension of the data.
      energy_hidden_sizes: The sizes of the hidden layers for the MLP that parameterizes the 
        energy function.
      proposal: A distribution over the data space of this model. Must support sample()
        and log_prob() although log_prob only needs to return a lower bound on the true
        log probability. If not supplied, then defaults to Gaussian.
    """
    self.K = K
    with tf.name_scope(name):
      self.energy_fn = functools.partial(
            mlp,
            layer_sizes=energy_hidden_sizes + [1],
            final_activation=None,
            name="energy_fn_mlp")
    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([data_dim], dtype=dtype))
    else:
      self.proposal = proposal
   
  def log_prob(self, data):
    batch_size = tf.shape(data)[0]

    proposal_samples = self.proposal.sample([batch_size, self.K])  #[batch_size, K, data_size]

    log_energy_target = tf.reshape(self.energy_fn(data), [batch_size])
    log_energy_proposal = tf.reshape(self.energy_fn(proposal_samples), [batch_size, self.K])

    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1) # [batch_size]

    #[batch_size]
    denom = tf.reduce_logsumexp(tf.stack([log_energy_target, proposal_lse], axis=-1), axis=1)
    denom -= tf.log(tf.to_float(self.K+1))

    lower_bound = self.proposal.log_prob(data) +  log_energy_target - denom
    return lower_bound

  def sample(self, sample_shape=[1]):
    shape = sample_shape + [self.K]
    proposal_samples = self.proposal.sample(shape) #[sample_shape, K, data_dim]
    log_energy = tf.reshape(self.energy_fn(proposal_samples), shape) #[sample_shape, K]
    indexes = tfd.Categorical(logits=log_energy).sample() #[sample_shape]
    #[sample_shape, data_dim]
    samples = tf.batch_gather(proposal_samples, tf.expand_dims(indexes, axis=-1)) 
    return samples
