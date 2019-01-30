import os
import gzip
from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class DatasetType(Enum):
  CONTINUOUS = 1
  BINARY = 2

DATASET_TYPES = {
    "nine_gaussians": DatasetType.CONTINUOUS,
    "raw_mnist": DatasetType.CONTINUOUS,
    "dynamic_mnist": DatasetType.BINARY,
}

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

DEFAULT_MNIST_PATH="data/mnist"

def get_raw_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(batch_size, split=split, binarized=None, 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def get_dynamic_mnist(batch_size, split="train", repeat=True, shuffle=True, initializable=False):
  return _get_mnist(batch_size, split=split, binarized="dynamic", 
          repeat=repeat, shuffle=shuffle, initializable=initializable)

def _get_mnist(batch_size, split="train", binarized=None, repeat=True, shuffle=True,
        initializable=False):
  if split == "train":
    im_path = os.path.join(DEFAULT_MNIST_PATH, "train-images-idx3-ubyte.gz")
    lb_path = os.path.join(DEFAULT_MNIST_PATH, "train-labels-idx1-ubyte.gz")
  elif split == "test":
    im_path = os.path.join(DEFAULT_MNIST_PATH, "t10k-images-idx3-ubyte.gz")
    lb_path = os.path.join(DEFAULT_MNIST_PATH, "t10k-labels-idx1-ubyte.gz")

  np_ims = _load_mnist_images(im_path)
  np_lbs = _load_mnist_labels(lb_path)
  mean = np.load(os.path.join(DEFAULT_MNIST_PATH, "train_mean.npy"))
  dataset = tf.data.Dataset.from_tensor_slices((np_ims, np_lbs))

  if binarized == "dynamic":
      dataset = dataset.map(lambda im, lb: (tfd.Bernoulli(logits=im).sample(), lb))
      
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

  ims, labels = itr.get_next()
  ims = tf.reshape(ims, [batch_size, 784])
  labels = tf.reshape(labels, [batch_size])
  return ims, labels, mean[tf.newaxis,:], itr

def _load_mnist_images(path):
  with gzip.open(path) as f:
    # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
    pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255

def _load_mnist_labels(path):
  with gzip.open(path) as f:
    # First 8 bytes are magic_number, n_labels
    integer_labels = np.frombuffer(f.read(), 'B', offset=8)
  return integer_labels
