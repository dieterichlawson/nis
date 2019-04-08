import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools
from . import base

def _expand_to_ta(x, length):
  x = tf.convert_to_tensor(x)
  if x.get_shape().ndims == 0:
    expanded = tf.tile(tf.reshape(x, [1]), [length])
  elif x.get_shape().ndims == 1:
    if x.get_shape()[0] == length:
      expanded = x
    else:
      expanded = tf.tile(tf.reshape(x,[1]), [length])
  ta = tf.TensorArray(dtype=x.dtype,
                      size=length,
                      dynamic_size=False,
                      clear_after_read=False,
                      infer_shape=True).unstack(expanded)
  return ta

def _hamiltonian_dynamics(x_0, momentum_0, energy_fn, T, step_size, temps):
  temperature_ta = _expand_to_ta(temps, T)
  t_0 = tf.constant(0, tf.int32, name="t_0")
  x_ta = tf.TensorArray(dtype=x_0.dtype, size=T, dynamic_size=False,
                      clear_after_read=False,infer_shape=True)
  momentum_ta = tf.TensorArray(dtype=momentum_0.dtype, size=T, dynamic_size=False,
                               clear_after_read=False,infer_shape=True)
  kinetic_energy_ta = tf.TensorArray(dtype=x_0.dtype, size=T, dynamic_size=False,
                                     clear_after_read=False,infer_shape=True)
  potential_energy_ta = tf.TensorArray(dtype=x_0.dtype, size=T, dynamic_size=False,
                                       clear_after_read=False,infer_shape=True)
  tas = [x_ta, momentum_ta, kinetic_energy_ta, potential_energy_ta]

  def _step(t, prev_x, prev_momentum, prev_energy_grad, tas):
    temp = temperature_ta.read(t)
    momentum_tilde = prev_momentum - (step_size/2.)*prev_energy_grad
    new_x = prev_x + step_size*momentum_tilde
    energy_at_new_x = tf.squeeze(energy_fn(new_x))
    grad_energy_at_new_x = tf.gradients(energy_at_new_x, new_x)[0]
    new_momentum = temp*(momentum_tilde - (step_size/2.)*grad_energy_at_new_x)
    kinetic_energy = tf.reduce_sum(tf.square(new_momentum)/2.)
    ta_updates = [new_x, new_momentum, kinetic_energy, energy_at_new_x]
    tas = [ta.write(t, z) for ta, z in zip(tas, ta_updates)]
    return (t+1, new_x, new_momentum, grad_energy_at_new_x, tas)

  def _predicate(t, *unused_args):
    return t < T

  grad_energy_0 = tf.gradients(energy_fn(x_0), x_0)[0]
  _, final_x, final_momentum, _, tas = tf.while_loop(
      _predicate,
      _step,
      loop_vars=(t_0, x_0, momentum_0, grad_energy_0, tas))
  xs, momentums, kes, pes = [t.stack() for t in tas]
  return final_x, final_momentum, xs, momentums, kes, pes 


def _reverse_hamiltonian_dynamics(x_T, momentum_T, energy_fn, T, step_size, temps):
  temperature_ta = _expand_to_ta(temps, T)
  t_T = tf.constant(T-1, tf.int32, name="t_T")
  x_ta = tf.TensorArray(dtype=x_T.dtype, size=T, dynamic_size=False,
                      clear_after_read=False,infer_shape=True)
  momentum_ta = tf.TensorArray(dtype=momentum_T.dtype, size=T, dynamic_size=False,
                               clear_after_read=False,infer_shape=True)
  kinetic_energy_ta = tf.TensorArray(dtype=x_T.dtype, size=T, dynamic_size=False,
                                     clear_after_read=False,infer_shape=True)
  potential_energy_ta = tf.TensorArray(dtype=x_T.dtype, size=T, dynamic_size=False,
                                       clear_after_read=False,infer_shape=True)
  tas = [x_ta, momentum_ta, kinetic_energy_ta, potential_energy_ta]

  def _step(t, next_x, next_momentum, next_energy_grad, tas):
    temp = temperature_ta.read(t)
    momentum_tilde = next_momentum/temp + (step_size/2.)*next_energy_grad
    prev_x = next_x - step_size*momentum_tilde
    energy_at_prev_x = tf.squeeze(energy_fn(prev_x))
    grad_energy_at_prev_x = tf.gradients(energy_at_prev_x, prev_x)[0]
    prev_momentum = momentum_tilde + (step_size/2.)*grad_energy_at_prev_x
    kinetic_energy = tf.reduce_sum(tf.square(prev_momentum)/2.)
    ta_updates = [prev_x, prev_momentum, kinetic_energy, energy_at_prev_x]
    new_tas = [ta.write(t, z) for ta, z in zip(tas, ta_updates)]
    return (t-1, prev_x, prev_momentum, grad_energy_at_prev_x, new_tas)

  def _predicate(t, *unused_args):
    return t >= 0

  grad_energy_T = tf.gradients(energy_fn(x_T), x_T)[0]
  _, x_0, momentum_0, _, tas= tf.while_loop(
      _predicate,
      _step,
      loop_vars=(t_T, x_T, momentum_T, grad_energy_T, tas))
  xs, momentums, kes, pes = [t.stack() for t in tas]
  return x_0, momentum_0, xs, momentums, kes, pes

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
      self.raw_alphas = tf.get_variable(name="raw_alpha",
                                        shape=[T],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(init_alpha),
                                        trainable=learn_temps)
      self.alphas = tf.math.sigmoid(self.raw_alphas)
      init_step_size = np.log(np.exp(init_step_size) - 1.)
      self.raw_step_size = tf.get_variable(name="raw_step_size",
                                           shape=[data_dim],
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(init_step_size),
                                           trainable=learn_stepsize)
      self.step_size = tf.math.softplus(self.raw_step_size)

    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2*data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([2*data_dim], dtype=dtype))
    else:
      self.proposal = proposal

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    q = self.q(data)
    rho_T = q.sample([num_samples])
    x_T = tf.tile(data[tf.newaxis,:,:], [num_samples, 1,1])
    x_0, rho_0, _, _, _, pes = _reverse_hamiltonian_dynamics(x_T, rho_T, self.energy_fn, self.T,
                                                             step_size=self.step_size, temps=self.alphas)
    tf.summary.histogram("energies", pes)
    tf.summary.scalar("min_energies", tf.reduce_min(pes))
    tf.summary.scalar("max_energies", tf.reduce_max(pes))
    log_p0 = self.proposal.log_prob(tf.concat([x_0, rho_0], axis=2))
    elbo = log_p0 - self.data_dim*tf.reduce_sum(tf.log(self.alphas)) - q.log_prob(rho_T)
    return tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(num_samples))

  def sample(self, sample_shape=[1]):
    x_and_rho = self.proposal.sample(sample_shape=sample_shape)
    x_0, rho_0 = tf.split(x_and_rho, 2, axis=-1)
    x_T, rho_T, xs, _, _, _ = _hamiltonian_dynamics(x_0, rho_0, self.energy_fn, self.T,
                                                    step_size=self.step_size, temps=self.alphas)
    return x_T#, rho_T, xs

