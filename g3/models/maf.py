import google3
import functools
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from google3.experimental.users.gjt.his.models import base

class MAF(object):

  def __init__(self,
               data_dim,
               hidden_sizes,
               flow_layers,
               proposal=None,
               data_mean=None,
               alpha=1e-6,
               dtype=tf.float32,
               name="maf"):
    """Creates a MAF model.

    Args:
      data_dim: The dimension of the data.
      hidden_sizes: The sizes of the hidden layers for the MLP that parameterizes the
        energy function.
      proposal: A distribution over the data space of this model. Must support sample()
        and log_prob() although log_prob only needs to return a lower bound on the true
        log probability. If not supplied, then defaults to Gaussian. Must be a tf distribution!
    """
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([data_dim], dtype=dtype))
    else:
      self.proposal = proposal


    bijectors = [tfb.AffineScalar(scale=256.),
                 tfb.AffineScalar(scale=1./(1. - alpha), shift=0.5*(1. - 1./(1.-alpha))),
                 tfb.Sigmoid()]
    for _ in range(flow_layers):
      bijectors.append(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=hidden_sizes)))
      bijectors.append(tfb.Permute(permutation=list(reversed(range(self.data_dim)))))
    self.maf = tfd.TransformedDistribution(
        distribution=self.proposal,
        bijector=tfb.Chain(bijectors[:-1]))


  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    return self.maf.log_prob(data)

  def sample(self, sample_shape=[1]):
    return self.maf.sample(sample_shape)
