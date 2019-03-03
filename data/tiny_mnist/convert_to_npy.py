import sys
import os
import gzip
import numpy as np

def load_mnist_images(path):
  with gzip.open(path) as f:
    # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
    pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255

def load_mnist_labels(path):
  with gzip.open(path) as f:
    # First 8 bytes are magic_number, n_labels
    integer_labels = np.frombuffer(f.read(), 'B', offset=8)
  return integer_labels

all_train_ims = load_mnist_images("train-ims.gz")
all_train_lbs = load_mnist_labels("train-lbs.gz")

train_one_ims = [im for im, lb in zip(all_train_ims, all_train_lbs) if lb == 1]

all_test_ims = load_mnist_images("test-ims.gz")
all_test_lbs = load_mnist_labels("test-lbs.gz")
test_one_ims = [im for im, lb in zip(all_test_ims, all_test_lbs) if lb == 1]

np.save("train", train_one_ims)
np.save("test", test_one_ims)
np.save("train_mean", np.mean(train_one_ims, axis=0))
