""" Default configuration """
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("num_classes", 4,       "Number of classes")
flags.DEFINE_integer("num_epochs", 10,       "Number of epochs to train")
flags.DEFINE_integer("snapshot_iters", 1000, "Number of iterations to make a checkpoint")
flags.DEFINE_integer("batch_size", 16,       "Batch size")
flags.DEFINE_boolean("is_training", True,    "True for training, False for testing [True]")
flags.DEFINE_boolean("batch_norm",  True,    "True for using batch norm [False]")
flags.DEFINE_boolean("conditioned", True,    "Transformation matrix conditioned in inputs [False]")
default_config = flags.FLAGS

class SizeContainer:
  def __init__(self):
    self.H_src, self.W_src, self.C_src = [224, 224,  3]   # aerial image size
    self.H_tar, self.W_tar, self.C_tar = [224, 1232, 3]   # ground image size
    self.before_transf = [17, 17]   # aerial probability map size
    self.after_transf  = [8,  40]   # ground probability map size

    self.image_aerial = [self.H_src, self.W_src]
    self.image_ground = [self.H_tar, self.W_tar]
