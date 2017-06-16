from tensorflow.python.ops import control_flow_ops
import models, misc, os, time, config
from random import shuffle
from scipy.misc import imread, imresize, imsave
import numpy as np
import tensorflow as tf

class CrossNet(object):
  def __init__(self, sess):
    self.sess        = sess
    self.szs         = config.SizeContainer()
    self.config      = config.default_config
    self.batch_size  = self.config.batch_size
    self.num_classes = self.config.num_classes
    self.conditioned = self.config.conditioned
    self.batch_norm  = self.config.batch_norm
    self.is_training = self.config.is_training

    self.log_dir  = misc.mkdir('outputs/logs/')
    self.ckpt_dir = misc.mkdir('outputs/ckpts/')
    self.dump_dir = misc.mkdir('outputs/dump/{}'.format('train' if self.is_training else 'deploy'))

    self.image_aerial_holder = tf.placeholder(tf.float32, [self.batch_size, None, None, self.szs.C_src])
    self.image_ground_holder = tf.placeholder(tf.float32, [self.batch_size, None, None, self.szs.C_tar])
    self.label_ground_holder = tf.placeholder(tf.int32,   [self.batch_size, None, None])

    self.build_model([
      self.image_aerial_holder,
      self.image_ground_holder,
      self.label_ground_holder],
      self.is_training)

  def load_data(self, image_list, image_dir=""):
    """Create a queue that outputs batches of images and labels
       label 0~3: [sky, bldg, road, tree]
    """
    self.data_names = []
    with open(image_list, 'r') as fid:
      for line in fid.readlines():
        names = [os.path.join(image_dir, name.strip()) for name in line.split(',')]
        keys = ['im_a', 'im_g', 'lb_g']
        self.data_names.append(dict(zip(keys, names)))
    shuffle(self.data_names)

    self.num_samples = len(self.data_names)
    misc.pprint('[*] load %d samples from "%s"' % (self.num_samples, image_list))

  def feed_dict_generator(self):
    for ib in xrange(0, self.num_samples, self.batch_size):
      image_aerial_batch = []
      image_ground_batch = []
      label_ground_batch = []
      for ix in xrange(self.batch_size):
        names = self.data_names[(ib+ix)%self.num_samples]
        im_a = imread(names['im_a'])
        im_g = imread(names['im_g'])
        lb_g = imread(names['lb_g'])
        im_a = misc.center_crop(im_a, self.szs.image_aerial)
        im_g = imresize(im_g, self.szs.image_ground)
        lb_g = imresize(lb_g, self.szs.image_ground, interp='nearest')
        image_aerial_batch.append(im_a)
        image_ground_batch.append(im_g)
        label_ground_batch.append(lb_g)

      feed_dict = {self.image_aerial_holder: np.array(image_aerial_batch),
          self.image_ground_holder: np.array(image_ground_batch),
          self.label_ground_holder: np.array(label_ground_batch)}
      yield feed_dict

  def build_model(self, data, is_training=True):
    raw_aerial, raw_ground, label_ground = data
    self.image_aerial = misc.preprocess_image(raw_aerial, self.szs.image_aerial)
    self.image_ground = misc.preprocess_image(raw_ground, self.szs.image_ground)
    self.prob_ground  = misc.preprocess_label(label_ground, self.num_classes, self.szs.after_transf)
    self.im_aerial = misc.proprocess_image(self.image_aerial)
    self.im_ground = misc.proprocess_image(self.image_ground)

    self.feat_aerial = models.pixelnet(self.image_aerial, self.num_classes, 
        is_training=is_training, batch_norm=self.batch_norm)
    misc.pprint(self.feat_aerial.get_shape().as_list()) # print the feature size

    if is_training:
      feat_aerial_small = self.feat_aerial
    else:
      feat_aerial_small = tf.image.resize_bilinear(self.feat_aerial, self.szs.before_transf)

    weights = models.compute_transfweights(self.szs.before_transf, self.szs.after_transf, 
        self.conditioned, is_training=is_training, batch_norm=self.batch_norm)
    self.feat_aerial2ground = models.transfnet(feat_aerial_small, weights, self.szs.after_transf)

    if is_training:
      self.merged     = tf.summary.merge_all()
      self.summarizer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

      with tf.name_scope("Loss"):
        self.loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          None, self.prob_ground, self.feat_aerial2ground))
        self.loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = self.loss_class + self.loss_reg

      with tf.name_scope("Optimizer"):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
          self.step = tf.Variable(0,name='global_step',trainable=False)
          self.optim = tf.train.AdamOptimizer(
            tf.train.exponential_decay(0.001,self.step,5000,.7,staircase=True) # not sure if this is necessary for Adam optimizer
            ).minimize(self.loss, global_step=self.step)

    self.saver = tf.train.Saver(max_to_keep=10, write_version=tf.train.SaverDef.V2)
    misc.pprint("[*] build model.")

    self.transfweights, self.transfbiases = tf.get_collection('transformer_weights')
    self.prob_aerial = tf.nn.softmax(self.feat_aerial)
    self.prob_aerial2ground = tf.nn.softmax(self.feat_aerial2ground)

    with tf.name_scope('Vis'):
      self.visual = [ \
                    self.image_aerial, self.image_ground,
                    tf.cast(self.prob_aerial,        tf.float32)/self.num_classes, 
                    tf.cast(self.prob_ground,        tf.float32)/self.num_classes,
                    tf.cast(self.prob_aerial2ground, tf.float32)/self.num_classes,
                    self.transfweights]

  def restore(self):
    ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    misc.pprint("[*] restore checkpoint from '%s'." % self.ckpt_dir)

  def save(self):
    self.saver.save(self.sess, '%s/model.ckpt' % self.ckpt_dir, global_step=self.step)

  def train_test(self):
    if self.is_training:
      tf.global_variables_initializer().run()
      num_epochs = self.config.num_epochs
    else:
      self.restore()
      num_epochs = 1

    for iEpoch in xrange(num_epochs):
      for feed_dict in self.feed_dict_generator():
        if self.is_training:
          tic = time.time()
          _, summary, loss, step = self.sess.run([self.optim, self.merged, self.loss, self.step], feed_dict)
          toc = time.time()
          print("[%d] [%06d] step: %d, loss: %03.5f, (%05.3f s)" 
              % (iEpoch, step*self.batch_size, step, loss, toc-tic))
        else:
          try: step += 1
          except: step = 1
          print "[deploy mode] step: {}".format(step)

        visual  = self.sess.run(self.visual, feed_dict)

        if step % 100 == 1 or not self.is_training:
          montage = misc.to_montage(visual)
          save_path = misc.mkdir_for_file('%s/%06d.jpg' % (self.dump_dir, step))
          imsave(save_path, montage)

        if self.is_training:
          if step % 50 == 1: self.summarizer.add_summary(summary, step)
          if step % self.config.snapshot_iters == 0: self.save()

    if self.is_training: self.save()
