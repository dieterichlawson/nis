import sys
import os
import gzip
import numpy as np

def load_mnist_images(path):
  with gzip.open(path) as f:
    # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
    pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255

#def load_mnist_labels(path):
#  with gzip.open(path) as f:
#    # First 8 bytes are magic_number, n_labels
#    integer_labels = np.frombuffer(f.read(), 'B', offset=8)
#  return integer_labels

all_train_ims = load_mnist_images("train-ims.gz")
train_ims = all_train_ims[:50000,:]
valid_ims = all_train_ims[50000:,:]
test_ims = load_mnist_images("test-ims.gz")
#all_train_lbs = load_mnist_labels("train-lbs.gz")
#test_lbs = load_mnist_labels("test-lbs.gz")

np.save("train", train_ims)
np.save("valid", valid_ims)
np.save("test", test_ims)
np.save("train_mean", np.mean(train_ims, axis=0))
#np.save("train_lbs", train_lbs)
#np.save("test_lbs", test_lbs)
