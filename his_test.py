import tensorflow as tf
import numpy as np
import base

class HISTest(tf.test.TestCase):

  def test_simple_energy_fn(self):
    x_0 = tf.zeros([1], dtype=tf.float32)
    rho_0 = tf.zeros([1], dtype=tf.float32)
    energy_fn = lambda x: -x
    T = 3
    step_size = 1.
    temps = 1.
    with self.test_session() as sess:
      _, _, xs, momentums, kes, pes = base._hamiltonian_dynamics(x_0, rho_0, energy_fn, T,
              step_size, temps)
      xs_out, momentums_out, kes_out, pes_out = sess.run([xs, momentums, kes, pes])
      # The momentum should go up by 1 each step (grad of energy is -1 everywhere).
      self.assertAllClose(np.squeeze(momentums_out), np.arange(1.,4.))
      # The position should be the cumulative sum of the momentum.
      self.assertAllClose(np.squeeze(xs_out), np.cumsum(np.arange(0.5,3.5)))
      hamiltonians = kes_out + pes_out
      # Check that the hamiltonian is preserved.
      self.assertAllClose(hamiltonians, [hamiltonians[0]] * hamiltonians.shape[0])

  def test_stepsize(self):
    x_0 = tf.zeros([1], dtype=tf.float32)
    rho_0 = tf.zeros([1], dtype=tf.float32)
    energy_fn = lambda x: -x
    T = 3
    step_size = 0.5
    temps = 1.
    with self.test_session() as sess:
      _, _, xs, momentums, kes, pes = base._hamiltonian_dynamics(x_0, rho_0, energy_fn, T,
              step_size, temps)
      xs_out, momentums_out, kes_out, pes_out = sess.run([xs, momentums, kes, pes])
      self.assertAllClose(np.squeeze(momentums_out), np.arange(0.5, 2., step=0.5))
      self.assertAllClose(np.squeeze(xs_out), [0.125, 0.5, 1.125])
      hamiltonians = kes_out + pes_out
      # Check that the hamiltonian is preserved.
      self.assertAllClose(hamiltonians, [hamiltonians[0]] * hamiltonians.shape[0])

  def test_temps(self):
    x_0 = tf.zeros([1], dtype=tf.float32)
    rho_0 = tf.zeros([1], dtype=tf.float32)
    energy_fn = lambda x: -x
    T = 3
    step_size = 1. 
    temps = [0.1, 0.01, 0.001]
    with self.test_session() as sess:
      _, _, xs, momentums, kes, pes = base._hamiltonian_dynamics(x_0, rho_0, energy_fn, T,
              step_size, temps)
      xs_out, momentums_out, kes_out, pes_out = sess.run([xs, momentums, kes, pes])
      true_momentums = np.array([0.1, 0.011, 0.001011])
      true_xs = np.array([0.5, 1.1, 1.611])
      self.assertAllClose(np.squeeze(momentums_out), [0.1, 0.011, 0.001011])
      self.assertAllClose(np.squeeze(xs_out), [0.5, 1.1, 1.611])
      hamiltonians = kes_out + pes_out
      true_hamiltonian = -true_xs + np.square(true_momentums)/2.
      # Check that the hamiltonian changes according to the energy taken out of the momentum.
      self.assertAllClose(hamiltonians, true_hamiltonian)

  def test_forward_reverse(self):
    self._check_forward_and_reverse_are_same(step_size=1., temps=1., T=10)
    self._check_forward_and_reverse_are_same(step_size=0.5, temps=1., T=10)
    self._check_forward_and_reverse_are_same(step_size=0.5, temps=np.arange(1., 0., step=-0.1,
        dtype=np.float32), T=10, rtol=1e-5, atol=1e-5)

  def _check_forward_and_reverse_are_same(self, step_size, temps, T, rtol=1e-6, atol=1e-6):
    x_0 = tf.zeros([1], dtype=tf.float32)
    rho_0 = tf.zeros([1], dtype=tf.float32)
    energy_fn = lambda x: -x
    with self.test_session() as sess:
      _, _, xs, momentums, kes, pes = base._hamiltonian_dynamics(
              x_0, rho_0, energy_fn, T, step_size, temps)
      rev_x_0, rev_momentum_0, rev_xs, rev_momentums, rev_kes, rev_pes = base._reverse_hamiltonian_dynamics(
              xs[-1], momentums[-1], energy_fn, T, step_size, temps)
      outs = sess.run([xs, rev_xs, momentums, rev_momentums, kes, rev_kes, pes, rev_pes, rev_x_0,
          rev_momentum_0])
      self.assertAllClose(outs[0][:-1], outs[1][1:], rtol=rtol, atol=atol) # position
      self.assertAllClose(outs[2][:-1], outs[3][1:], rtol=rtol, atol=atol) # momentum
      self.assertAllClose(outs[4][:-1], outs[5][1:], rtol=rtol, atol=atol) # kinetic energy
      self.assertAllClose(outs[6][:-1], outs[7][1:], rtol=rtol, atol=atol) # potential energy

if __name__ == "__main__":
  tf.test.main()
