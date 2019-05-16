import google3
import os
import gzip
from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import pdb
tfd = tfp.distributions


def get_nine_gaussians(batch_size, scale=0.1, spacing=1.0):
  """Creates a mixture of 9 2-D gaussians on a 3x3 grid centered at 0."""
  components = []
  for i in [-spacing, 0., spacing]:
    for j in [-spacing, 0., spacing]:
      loc = tf.constant([i, j], dtype=tf.float32)
      scale = tf.ones_like(loc) * scale
      components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale))

  dist = tfd.Mixture(
      cat=tfd.Categorical(probs=tf.ones([9], dtype=tf.float32) / 9.),
      components=components)
  batch = dist.sample(batch_size)
  return batch


CNS_PATH = "/cns/lu-d/home/gjt"
TINY_MNIST_PATH = "data/tiny_mnist"
MNIST_PATH = "data/mnist"
STATIC_BINARIZED_MNIST_PATH = "data/static_binarized_mnist"


def get_raw_mnist(batch_size,
                  split="train",
                  repeat=True,
                  shuffle=True,
                  initializable=False):
  return _get_mnist(
      MNIST_PATH,
      batch_size,
      split=split,
      binarized=None,
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable)


def get_jittered_mnist(batch_size,
                       split="train",
                       repeat=True,
                       shuffle=True,
                       initializable=False):
  return _get_mnist(
      MNIST_PATH,
      batch_size,
      split=split,
      binarized=None,
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable,
      jitter=True)


def get_tiny_mnist(batch_size,
                   split="train",
                   repeat=True,
                   shuffle=True,
                   initializable=False,
                   binarized=None):
  return _get_mnist(
      TINY_MNIST_PATH,
      batch_size,
      split=split,
      binarized=binarized,
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable)


def get_dynamic_mnist(batch_size,
                      split="train",
                      repeat=True,
                      shuffle=True,
                      initializable=False):
  return _get_mnist(
      MNIST_PATH,
      batch_size,
      split=split,
      binarized="dynamic",
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable)


def get_static_mnist(batch_size,
                     split="train",
                     repeat=True,
                     shuffle=True,
                     initializable=False):
  return _get_mnist(
      STATIC_BINARIZED_MNIST_PATH,
      batch_size,
      split=split,
      binarized="static",
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable)


def _get_mnist(data_dir,
               batch_size,
               split="train",
               binarized=None,
               repeat=True,
               shuffle=True,
               initializable=False,
               jitter=False):
  path = os.path.join(CNS_PATH, data_dir, split + ".npy")
  with tf.io.gfile.GFile(path, "rb") as f:
    np_ims = np.load(f)
  # Always load the train mean, no matter what split.
  mean_path = os.path.join(CNS_PATH, data_dir, "train_mean.npy")
  with tf.io.gfile.GFile(mean_path, "rb") as f:
    mean = np.load(f).astype(np.float32)
  dataset = tf.data.Dataset.from_tensor_slices(np_ims)

  if binarized == "dynamic":
    dataset = dataset.map(lambda im: tfd.Bernoulli(probs=im).sample())
  elif jitter:
    # Add uniform dequantization jitter
    def jitter_im(im):
      jitter_noise = tfd.Uniform(
          low=tf.zeros_like(im), high=tf.ones_like(im)).sample()
      jittered_im = im * 255. + jitter_noise
      return jittered_im

    dataset = dataset.map(jitter_im)
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
  return ims, mean[tf.newaxis, :], itr

# CelebA dataset
CELEBA_IMAGE_SIZE = 64


def _preprocess(sample, crop_width=80, image_size=CELEBA_IMAGE_SIZE):
  """Output images are in [0, 255]."""
  image_shape = sample["image"].shape
  crop_slices = [
      slice(w // 2 - crop_width, w // 2 + crop_width) for w in image_shape[:2]
  ] + [slice(None)]
  image_cropped = sample["image"][crop_slices]
  image_resized = tf.image.resize_images(image_cropped, [image_size] * 2)
  x = tf.to_float(image_resized)
  return x


def get_jittered_celeba(batch_size, split="train"):
  datasets = tfds.load("celeb_a")
  train_data = datasets["train"].map(_preprocess)
  cur_sum, n = train_data.reduce((0., 0),
                                 lambda (cur_sum, n), x: (cur_sum + x, n + 1))
  train_mean = cur_sum / tf.to_float(n)
  train_mean.set_shape(train_data.output_shapes)
  data = datasets[split].map(_preprocess)

  # Add jitter
  def jitter_im(im):
    jitter_noise = tfd.Uniform(
        low=tf.zeros_like(im), high=tf.ones_like(im)).sample()
    jittered_im = im + jitter_noise
    return jittered_im
  data = data.map(jitter_im)

  itr = (data.batch(batch_size)
         .prefetch(tf.data.experimental.AUTOTUNE)
         .shuffle(1024)
         .cache()
         .repeat()
         .make_one_shot_iterator())
  ims = itr.get_next()
  return ims, train_mean[None, :], itr
