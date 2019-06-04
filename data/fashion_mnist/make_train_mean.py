import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

d = (tfds.load(name='fashion_mnist', split=tfds.Split.TRAIN)
     .map(lambda x: tf.to_float(x['image'])/255.))
def reduce_fn(a, x):
  cur_sum, n = a
  return (cur_sum + x, n + 1)
total_sum, n = d.reduce((0., 0), reduce_fn)
mean = total_sum / tf.to_float(n)

sess = tf.Session()
m = sess.run(mean)

np.save("train_mean", m)
