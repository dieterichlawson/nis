r"""XManager launcher for main_loop.py.

Instructions:

/google/data/ro/teams/dmgi/google_xmanager.par launch \
xm_train_launcher.py \
 -- \
 --cell=lu \
 --xm_resource_alloc=user:gjt \
 --xm_skip_launch_confirmation --trial=7

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass

from absl import app
from absl import flags

from google3.learning.deepmind.python.adhoc_import import binary_import
from google3.learning.deepmind.xmanager import hyper
import google3.learning.deepmind.xmanager2.client.google as xm

# Use adhoc import when running directly from the dm_python interpreter
with binary_import.AutoGoogle3():
  from google3.learning.brain.frameworks.xmanager import xm_helper  # pylint: disable=g-import-not-at-top

FLAGS = flags.FLAGS

flags.DEFINE_string('cell', 'lu', 'Cell on which launch the jobs.')
flags.DEFINE_integer('priority', 200, 'Priority to which launch the job.')
flags.DEFINE_integer('trial', 1, 'The trial number')
flags.DEFINE_string('model_dir',
                    '/cns/{cell}-d/home/{user}/experiments/xm/his_gjt/{trial}',
                    'Experiment path.')
flags.DEFINE_string('exp_type', 'snis_vs_lars',
                    'Which hyperparam sweep to run.')


def build_experiment():
  """Create the jobs/config and return the constructed experiment."""

  # ====== Argument creation ======
  model_dir = FLAGS.model_dir.format(
      cell=FLAGS.cell,
      user=getpass.getuser(),
      trial=FLAGS.trial,
  )

  # ====== Jobs and runtime creation ======

  # Job: worker
  requirements = xm.Requirements(gpu_types=[xm.GpuType.V100],)
  runtime_worker = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
      requirements=requirements,
  )
  exec_worker = xm.BuildTarget(
      '//experimental/users/gjt/his:mnist',
      name='worker',
      args=dict(
          gfs_user=FLAGS.gfs_user,
          logdir=model_dir,
          mode='train',
      ),
      platform=xm.Platform.GPU,
      runtime=runtime_worker,
  )

  # Job: eval
  runtime_eval = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
  )
  exec_eval = xm.BuildTarget(
      '//experimental/users/gjt/his:mnist',
      name='eval',
      args=dict(
          gfs_user=FLAGS.gfs_user,
          logdir=model_dir,
          mode='eval',
          split='train,valid,test',
          num_iwae_samples='1,1,1000',
      ),
      platform=xm.Platform.GPU,  # Do we need GPU for eval?
      runtime=runtime_eval,
  )

  # ====== Executable experiment creation ======
  list_executables = []
  list_executables.append(xm_helper.build_single_job(exec_worker))
  list_executables.append(xm_helper.build_single_job(exec_eval))

  experiment = xm.ParallelExecutable(list_executables, name='his_service')

  # Build experiments
  hyper_parameters = {}

  # SNIS vs LARS
  hyper_parameters['snis_vs_lars'] = hyper.product([
      hyper.chainit([
          hyper.product([
              hyper.fixed('proposal', 'gaussian', length=1),
              hyper.fixed('model', 'bernoulli_vae', length=1),
          ]),
          hyper.product([
              hyper.fixed('proposal', 'bernoulli_vae', length=1),
              hyper.fixed('model', 'nis', length=1),
          ]),
          hyper.product([
              hyper.fixed('proposal', 'nis', length=1),
              hyper.fixed('model', 'bernoulli_vae', length=1),
          ]),
      ]),
      hyper.sweep('run', hyper.discrete([0])),
      hyper.fixed('dataset', 'static_mnist', length=1),
      hyper.fixed('reparameterize_proposal', False, length=1),
      hyper.fixed('anneal_kl_step', 100000, length=1),
  ])

  # Continuous comparisons: HIS, NIS, VAE
  hyper_parameters['continuous'] = hyper.product([
      hyper.chainit([
          hyper.product([
              hyper.fixed('proposal', 'gaussian', length=1),
              hyper.fixed('model', 'gaussian_vae', length=1),
          ]),
          hyper.product([
              hyper.fixed('proposal', 'gaussian_vae', length=1),
              hyper.fixed('model', 'nis', length=1),
          ]),
          hyper.product([
              hyper.fixed('proposal', 'gaussian', length=1),
              hyper.fixed('model', 'hisvae', length=1),
              hyper.sweep('his_T', hyper.discrete([5, 10, 15])),
          ]),
      ]),
      hyper.sweep('run', hyper.discrete([0])),
      hyper.fixed('dataset', 'jittered_mnist', length=1),
      hyper.fixed('reparameterize_proposal', True, length=1),
      hyper.fixed('squash', True, length=1),
  ])

  hyper_parameters['celeba'] = hyper.product([
      hyper.chainit([
          hyper.product([
              hyper.fixed('proposal', 'gaussian', length=1),
              hyper.fixed('model', 'conv_gaussian_vae', length=1),
          ]),
      ]),
      hyper.sweep('run', hyper.discrete([0])),
      hyper.fixed('dataset', 'jittered_celeba', length=1),
      hyper.fixed('reparameterize_proposal', True, length=1),
      hyper.fixed('squash', True, length=1),
      hyper.fixed('latent_dim', 16, length=1),
      hyper.fixed('batch_size', 36, length=1),
  ])

  experiment = xm.ParameterSweep(experiment, hyper_parameters[FLAGS.exp_type])
  experiment = xm.WithTensorBoard(experiment, model_dir)

  return experiment


def main(_):
  """Launch the experiment using the arguments from the command line."""
  description = xm.ExperimentDescription(
      'HIS - trial=%d' % FLAGS.trial, tags=['his'])
  experiment = build_experiment()
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
