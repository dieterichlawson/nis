import os
import gzip
import numpy as np
import tensorflow as tf

DEFAULT_MNIST_PATH="data/mnist"

def get_mnist(batch_size, split="train"):
  if split == "train":
    im_path = os.path.join(DEFAULT_MNIST_PATH, "train-images-idx3-ubyte.gz")
    lb_path = os.path.join(DEFAULT_MNIST_PATH, "train-labels-idx1-ubyte.gz")
  elif split == "test":
    im_path = os.path.join(DEFAULT_MNIST_PATH, "t10k-images-idx3-ubyte.gz")
    lb_path = os.path.join(DEFAULT_MNIST_PATH, "t10k-labels-idx1-ubyte.gz")

  np_ims = get_images(im_path)
  np_lbs = get_labels(lb_path)
  mean = np.load(os.path.join(DEFAULT_MNIST_PATH, "train_mean.npy"))
  dataset = tf.data.Dataset.from_tensor_slices((np_ims, np_lbs))

  
  if split == "train":
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1024)
  itr = dataset.make_one_shot_iterator()
  ims, labels = itr.get_next()
    ims = tf.reshape(ims, [batch_size, 784])
  labels = tf.reshape(labels, [batch_size])
  return ims, labels, mean[tf.newaxis,:]

def get_images(path):
  with gzip.open(path) as f:
    # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
    pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255

def get_labels(path):
  with gzip.open(path) as f:
    # First 8 bytes are magic_number, n_labels
    integer_labels = np.frombuffer(f.read(), 'B', offset=8)
  return integer_labels
