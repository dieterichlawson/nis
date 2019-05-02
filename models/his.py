import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools
from . import base


class HIS(object):

  def __init__(self,
               T,
               data_dim,
               energy_hidden_sizes,
               q_hidden_sizes,
               proposal=None,
               data_mean=None,
               init_alpha=1.,
               init_step_size=0.01,
               learn_temps=False,
               learn_stepsize=False,
               scale_min=1e-5,
               dtype=tf.float32,
               name="his"):
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.T = T
    self.energy_fn = functools.partial(
          base.mlp,
          layer_sizes=energy_hidden_sizes + [1],
          final_activation=None,
          name="%s/energy_fn_mlp" % name)
    self.q = functools.partial(
          base.conditional_normal,
          data_dim=data_dim,
          hidden_sizes=q_hidden_sizes,
          scale_min=scale_min,
          bias_init=None,
          truncate=False,
          squash=False,
          name="%s/q" % name)
    eps = 0.0001
    init_alpha = -np.log(1./init_alpha - 1. + eps)
    with tf.name_scope(name):
      self.raw_alphas = [tf.get_variable(name="raw_alpha_%d" % t,
                                        shape=[],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(init_alpha),
                                        trainable=learn_temps)
                         for t in range(T)]
      self.log_alphas = [-tf.nn.softplus(-raw_alpha) for raw_alpha in self.raw_alphas]
      self.log_alphas = [-tf.reduce_sum(self.log_alphas)] + self.log_alphas
      init_step_size = np.log(np.exp(init_step_size) - 1.)
      self.raw_step_size = tf.get_variable(name="raw_step_size",
                                           shape=[data_dim],
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(init_step_size),
                                           trainable=learn_stepsize)
      self.step_size = tf.math.softplus(self.raw_step_size)

    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([data_dim], dtype=dtype))
    else:
      self.proposal = proposal
    self.momentum_proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                        scale_diag=tf.ones([data_dim], dtype=dtype))

    # Combine energy and proposal into full potential function.
    self.hamiltonian_potential = lambda x: (tf.squeeze(self.energy_fn(x))
                                            - self.proposal.log_prob(x))

  def _grad_hamiltonian_potential(self, x):
    potential = self.hamiltonian_potential(x)
    return tf.gradients(potential, x)[0]

  def _hamiltonian_dynamics(self, x, momentum, alphas=None):
    if alphas is None:
      alphas = map(tf.exp, self.log_alphas)

    momentum *= alphas[0]
    grad_energy = self._grad_hamiltonian_potential(x)
    for t in range(1, self.T+1):
      momentum -= self.step_size/2.*grad_energy
      x += self.step_size * momentum
      grad_energy = self._grad_hamiltonian_potential(x)
      momentum -= self.step_size/2.*grad_energy
      momentum *= alphas[t]
    return x, momentum

  def _reverse_hamiltonian_dynamics(self, x, momentum):
    alphas = [tf.exp(-log_alpha) for log_alpha in self.log_alphas]
    alphas.reverse()
    x, momentum = self._hamiltonian_dynamics(x, -momentum, alphas)
    return x, -momentum

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_joint(self, x_T, rho_T):
    x_0, rho_0 = self._reverse_hamiltonian_dynamics(x_T, rho_T)
    return self.proposal.log_prob(x_0) + self.momentum_proposal.log_prob(rho_0)

  def _log_prob(self, data, num_samples=1):
    q = self.q(data)
    rho_T = q.sample([num_samples])
    x_T = tf.tile(data[tf.newaxis,:,:], [num_samples, 1,1])

    #tf.summary.histogram("energies", pes)
    #tf.summary.scalar("min_energies", tf.reduce_min(pes))
    #tf.summary.scalar("max_energies", tf.reduce_max(pes))
    log_joint = self._log_joint(x_T, rho_T)
    elbo = log_joint - q.log_prob(rho_T)
    return tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(num_samples))

  def sample(self, sample_shape=[1]):
    x_0, rho_0 = self.proposal.sample(sample_shape=sample_shape), self.momentum_proposal.sample(sample_shape=sample_shape)
    x_T, _ = _hamiltonian_dynamics(x_0, rho_0)
    return x_T

