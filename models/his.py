import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools
from . import base

import pdb

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
               squash=False,
               squash_eps=1e-6,
               dtype=tf.float32,
               name="his"):
    if squash:
      bijectors = [tfp.bijectors.AffineScalar(scale=256.),
                   tfp.bijectors.AffineScalar(shift=-squash_eps/2.,
                                              scale=(1. + squash_eps)),
                   tfp.bijectors.Sigmoid(),
                   ]
      self.squash = tfp.bijectors.Chain(bijectors)
    else:
      self.squash = None

    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean

      if squash:
        self.unsquashed_data_mean = self.squash.inverse(data_mean)
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
      tf.summary.scalar("his_step_size", tf.reduce_mean(self.step_size))
      [tf.summary.scalar("his_alpha/alpha_%d" % t, tf.exp(self.log_alphas[t]))
       for t in range(len(self.log_alphas))]


    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([data_dim], dtype=dtype))
    else:
      self.proposal = proposal
    self.momentum_proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                        scale_diag=tf.ones([data_dim], dtype=dtype))

  def hamiltonian_potential(self, x):
    pdb.set_trace()
    return tf.squeeze(self.energy_fn(x), axis=-1) - self.proposal.log_prob(x)

  def _grad_hamiltonian_potential(self, x):
    potential = self.hamiltonian_potential(x)
    return tf.gradients(potential, x)[0]

  def _hamiltonian_dynamics(self, x, momentum, alphas=None):
    if alphas is None:
      alphas = [tf.exp(log_alpha) for log_alpha in self.log_alphas]

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
    if self.squash is not None:
      x_T = self.squash.inverse(data)
      x_T -= self.unsquashed_data_mean
    else:
      x_T = data

    q = self.q(data)  # Maybe better to work w/ the untransformed data?
    rho_T = q.sample([num_samples])
    x_T = tf.tile(x_T[tf.newaxis,:,:], [num_samples, 1,1])

    #tf.summary.histogram("energies", pes)
    #tf.summary.scalar("min_energies", tf.reduce_min(pes))
    #tf.summary.scalar("max_energies", tf.reduce_max(pes))
    log_joint = self._log_joint(x_T, rho_T)
    elbo = log_joint - q.log_prob(rho_T)
    if self.squash is not None:
      elbo += tf.tile(self.squash.inverse_log_det_jacobian(data, event_ndims=1)[None, :], [num_samples, 1])
    return tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(num_samples))

  def sample(self, sample_shape=[1]):
    x_0, rho_0 = self.proposal.sample(sample_shape=sample_shape), self.momentum_proposal.sample(sample_shape=sample_shape)
    x_T, _ = self._hamiltonian_dynamics(x_0, rho_0)
    if self.squash is not None:
      x_T += self.unsquashed_data_mean
      x_T = self.squash.forward(x_T)
    return x_T


# Copied implementation of HIS, will abstract out relevant components
# afterwards.
class HISVAE(object):

  def __init__(self,
               T,
               latent_dim,
               data_dim,
               energy_hidden_sizes,
               q_hidden_sizes,
               decoder_hidden_sizes,
               proposal=None,
               data_mean=None,
               init_alpha=1.,
               init_step_size=0.01,
               learn_temps=False,
               learn_stepsize=False,
               scale_min=1e-5,
               squash=False,
               squash_eps=1e-6,
               decoder_nn_scale=False,
               dtype=tf.float32,
               name="hisvae"):
    if squash:
      bijectors = [tfp.bijectors.AffineScalar(scale=256.),
                   tfp.bijectors.AffineScalar(shift=-squash_eps/2.,
                                              scale=(1. + squash_eps)),
                   tfp.bijectors.Sigmoid(),
                   ]
      self.squash = tfp.bijectors.Chain(bijectors)
    else:
      self.squash = None

    self.latent_dim = latent_dim
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean

      if squash:
        self.unsquashed_data_mean = self.squash.inverse(data_mean)
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
          data_dim=data_dim + latent_dim,
          hidden_sizes=q_hidden_sizes,
          scale_min=scale_min,
          bias_init=None,
          truncate=False,
          squash=False,
          name="%s/q" % name)
    self.decoder = functools.partial(
        base.conditional_normal,
        data_dim=data_dim,
        hidden_sizes=decoder_hidden_sizes,
        scale_min=scale_min,
        nn_scale=decoder_nn_scale,
        #bias_init=data_mean,  # TODO: FIX THIS
        truncate=False,
        squash=False,
        name="%s/decoder" % name)

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
      tf.summary.scalar("his_step_size", tf.reduce_mean(self.step_size))
      [tf.summary.scalar("his_alpha/alpha_%d" % t, tf.exp(self.log_alphas[t]))
       for t in range(len(self.log_alphas))]


    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([latent_dim], dtype=dtype),
                                                 scale_diag=tf.ones([latent_dim], dtype=dtype))
    else:
      self.proposal = proposal
    self.momentum_proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                        scale_diag=tf.ones([data_dim], dtype=dtype))


  def hamiltonian_potential(self, x, z, p_x_given_z):
    return (tf.squeeze(self.energy_fn(tf.concat([x, z], axis=-1)), axis=-1)
            - p_x_given_z.log_prob(x))

  def _grad_hamiltonian_potential(self, x, z, p_x_given_z):
    potential = self.hamiltonian_potential(x, z, p_x_given_z)
    return tf.gradients(potential, x)[0]

  def _hamiltonian_dynamics(self, x, momentum, z, p_x_given_z, alphas=None):
    if alphas is None:
      alphas = [tf.exp(log_alpha) for log_alpha in self.log_alphas]

    momentum *= alphas[0]
    grad_energy = self._grad_hamiltonian_potential(x, z, p_x_given_z)
    for t in range(1, self.T+1):
      momentum -= self.step_size/2.*grad_energy
      x += self.step_size * momentum
      grad_energy = self._grad_hamiltonian_potential(x, z, p_x_given_z)
      momentum -= self.step_size/2.*grad_energy
      momentum *= alphas[t]
    return x, momentum

  def _reverse_hamiltonian_dynamics(self, x, momentum, z, p_x_given_z):
    alphas = [tf.exp(-log_alpha) for log_alpha in self.log_alphas]
    alphas.reverse()
    x, momentum = self._hamiltonian_dynamics(x, -momentum, z, p_x_given_z, alphas)
    return x, -momentum

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_joint(self, x_T, rho_T, z, p_x_given_z):
    x_0, rho_0 = self._reverse_hamiltonian_dynamics(x_T, rho_T, z, p_x_given_z)
    return p_x_given_z.log_prob(x_0) + self.momentum_proposal.log_prob(rho_0)

  def _log_prob(self, data, num_samples=1):
    if self.squash is not None:
      x_T = self.squash.inverse(data)
      x_T -= self.unsquashed_data_mean
    else:
      x_T = data

    q = self.q(data)  # Maybe better to work w/ the untransformed data?
    rho_T_and_z = q.sample([num_samples])
    rho_T = rho_T_and_z[:, :, :self.data_dim]
    z = rho_T_and_z[:, :, self.data_dim:]
    p_x_given_z = self.decoder(z)

    x_T = tf.tile(x_T[tf.newaxis,:,:], [num_samples, 1,1])

    log_joint = self._log_joint(x_T, rho_T, z, p_x_given_z)
    elbo = self.proposal.log_prob(z) + log_joint - q.log_prob(rho_T_and_z)
    if self.squash is not None:
      elbo += tf.tile(self.squash.inverse_log_det_jacobian(data, event_ndims=1)[None, :], [num_samples, 1])
    return tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(num_samples))

  def sample(self, sample_shape=[1]):
    z = self.proposal.sample(sample_shape)
    p_x_given_z = self.decoder(z)
    x_0 = p_x_given_z.sample()
    rho_0 = self.momentum_proposal.sample(sample_shape=sample_shape)
    x_T, _ = self._hamiltonian_dynamics(x_0, rho_0, z, p_x_given_z)

    initial_potential = self.hamiltonian_potential(x_0, z, p_x_given_z)
    final_potential = self.hamiltonian_potential(x_T, z, p_x_given_z)
    tf.summary.histogram("initial_potential", initial_potential)
    tf.summary.histogram("diff_potential", final_potential - initial_potential)
    final_energy = tf.squeeze(self.energy_fn(tf.concat([x_T, z], axis=-1)), axis=-1)
    tf.summary.histogram("final_energy", final_energy)

    if self.squash is not None:
      x_T += self.unsquashed_data_mean
      x_T = self.squash.forward(x_T)
    return x_T

