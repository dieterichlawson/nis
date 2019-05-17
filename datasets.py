import os
import gzip
from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import pdb

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

TINY_MNIST_PATH="data/tiny_mnist"
MNIST_PATH="data/mnist"
STATIC_BINARIZED_MNIST_PATH = "data/static_binarized_mnist"
FASHION_MNIST_PATH="data/fashion_mnist"

def get_raw_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(MNIST_PATH, batch_size, split=split, binarized=None, 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def get_jittered_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(MNIST_PATH, batch_size, split=split, binarized=None,
          repeat=repeat, shuffle=shuffle, initializable=initializable, jitter=True)

def get_tiny_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False,
        binarized=None):
  return _get_mnist(TINY_MNIST_PATH, batch_size, split=split, binarized=binarized, 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def get_dynamic_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(MNIST_PATH, batch_size, split=split, binarized="dynamic", 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def get_static_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(STATIC_BINARIZED_MNIST_PATH, batch_size, split=split, binarized="static", 
          repeat=repeat, shuffle=shuffle, initializable=initializable)


def _get_mnist(data_dir, batch_size, split="train", binarized=None, repeat=True, shuffle=True,
        initializable=False, jitter=False):
  path = os.path.join(data_dir, split + ".npy")
  np_ims = np.load(path)
  # Always load the train mean, no matter what split.
  mean = np.load(os.path.join(data_dir, "train_mean.npy")).astype(np.float32)
  dataset = tf.data.Dataset.from_tensor_slices(np_ims)

  if binarized == "dynamic":
    dataset = dataset.map(lambda im: tfd.Bernoulli(probs=im).sample())
  elif jitter:
    # Add uniform dequantization jitter
    def jitter_im(im):
      jitter_noise = tfd.Uniform(low=tf.zeros_like(im), high=tf.ones_like(im)).sample()
      jittered_im = im * 255. + jitter_noise
      return jittered_im

    dataset = dataset.map(lambda im: jitter_im(im))
    mean *= 255.

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

def get_fashion_mnist(batch_size, split="train", binarized=True, repeat=True, shuffle=True, 
                      initializable=False):
  dataset = tfds.load(name='fashion_mnist', split=splits[split])
  # Strip off the labels.
  dataset = dataset.map(lambda x: x['image'])
  if binarized:
    dataset = dataset.map(lambda im: tfd.Bernoulli(probs=im).sample())

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
  mean = np.load(os.path.join(FASHION_MNIST_PATH, "train_mean.npy")).astype(np.float32)
  return ims, mean, itr
