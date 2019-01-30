import os
import gzip
from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def get_nine_gaussians(batch_size, scale=0.1, spacing=1.0):
  """Creates a mixture of 9 2-D gaussians on a 3x3 grid centered at 0."""
  components = []
  for i in [-spacing, 0. , spacing]:
    for j in [-spacing, 0. , spacing]:
      loc = tf.constant([i,j], dtype=tf.float32)
      scale = tf.ones_like(loc)*scale
      components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale))

  dist = tfd.Mixture(
    cat=tfd.Categorical(probs=tf.ones([9], dtype=tf.float32)/9.),
    components=components)
  batch = dist.sample(batch_size)
  return batch

MNIST_PATH="data/mnist"
STATIC_BINARIZED_MNIST_PATH = "data/static_binarized_mnist"

def get_raw_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(batch_size, split=split, binarized=None, 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def get_dynamic_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(batch_size, split=split, binarized="dynamic", 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def get_static_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(batch_size, split=split, binarized="static", 
          repeat=repeat, shuffle=shuffle, initializable=initializable)


def _get_mnist(batch_size, split="train", binarized=None, repeat=True, shuffle=True,
        initializable=False):
  if binarized == "static":
    data_dir = STATIC_BINARIZED_MNIST_PATH
  elif binarized == "dynamic" or binarized == None:
    data_dir = MNIST_PATH

  path = os.path.join(data_dir, split + ".npy")
  np_ims = np.load(path)
  mean = np.load(os.path.join(data_dir, "train_mean.npy"))
  dataset = tf.data.Dataset.from_tensor_slices(np_ims)

  if binarized == "dynamic":
      dataset = dataset.map(lambda im: tfd.Bernoulli(logits=im).sample())
      
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(1024)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(128)

  if initializable:
    itr = dataset.make_initializable_iterator()
  else:
    itr = dataset.make_one_shot_iterator()

  ims = itr.get_next()
  true_batch_size = tf.shape(ims)[0]
  ims = tf.reshape(ims, [true_batch_size, 784])
  return ims, mean[tf.newaxis,:], itr
