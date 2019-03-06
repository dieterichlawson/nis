import os
import glob
import imageio
import tensorflow as tf

tf.app.flags.DEFINE_string("logdir", "/tmp/nis", "Directory containing Tensorboard.")
tf.app.flags.DEFINE_string("outdir", ".", "Directory to output images.")
tf.app.flags.DEFINE_string("tag", "image", "Tag name for images to extract.")

FLAGS = tf.app.flags.FLAGS

def save_images_from_event(logdir, tag, output_dir):
    assert(os.path.isdir(output_dir))
    assert(os.path.isdir(logdir))
    event_paths = sorted(glob.glob(os.path.join(logdir, "event*")))
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        for fn in event_paths:
          for e in tf.train.summary_iterator(fn):
              for v in e.summary.value:
                  if v.tag == tag:
                      im = im_tf.eval({image_str: v.image.encoded_image_string})
                      output_fn = os.path.realpath('{}/image_{:010d}.png'.format(output_dir, e.step))
                      imageio.imwrite(output_fn, im)

def main(unused_argv):
  save_images_from_event(FLAGS.logdir, FLAGS.tag, FLAGS.outdir)

if __name__ == "__main__":
  tf.app.run(main)
