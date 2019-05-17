import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

d = tfds.load(name='fashion_mnist', split=tfds.Split.TRAIN)
mean = d.reduce(0., lambda a,x: a+tf.cast(x['image'], tf.float32))
mean /= 50000. # there are 50,000  training examples
sess = tf.Session()
m = sess.run(mean)

np.save("train_mean", m)
