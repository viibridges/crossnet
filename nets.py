import tensorflow as tf
import contextlib
slim = tf.contrib.slim

batch_norm_params = {
  'decay': 0.9, 'epsilon': 0.001,
  'updates_collections': tf.GraphKeys.UPDATE_OPS,
  'variables_collections': {
      'beta': None, 'gamma': None,
      'moving_mean': ['moving_vars'],
      'moving_variance': ['moving_vars'],
  }
}

def vgg_arg_scope(batch_norm=True, weight_decay=0.0005):
  if batch_norm: 
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=batch_norm_params,
      weights_regularizer=slim.l2_regularizer(weight_decay),
      biases_initializer=tf.zeros_initializer) as arg_sc:
    return arg_sc

def base_arg_scope(is_training=True, batch_norm=True, weight_decay=.0005):
  return contextlib.nested(
      slim.arg_scope(vgg_arg_scope(batch_norm=batch_norm, weight_decay=weight_decay)), 
      slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training),
      slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding='VALID'))

def vgg_like(inputs):
  base_dim = 64 
  end_points = []
  with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
    net = slim.repeat(inputs, 2, slim.conv2d, base_dim, [3, 3], scope='conv1')
    end_points.append(net)
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, base_dim*2, [3, 3], scope='conv2')
    end_points.append(net)
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, base_dim*4, [3, 3], scope='conv3')
    end_points.append(net)
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, base_dim*8, [3, 3], scope='conv4')
    end_points.append(net)
  return net, end_points
