import tensorflow as tf
import contextlib, ops, nets
import numpy as np
slim = tf.contrib.slim

def pixelnet_convs(inputs, num_class, is_training=True, reuse=False):
  num_batch = tf.shape(inputs)[0]
  height = tf.shape(inputs)[1]
  width = tf.shape(inputs)[2]

  with tf.variable_scope('vgg_16', reuse=reuse):
    net, hyperfeats = nets.vgg_like(inputs)
    tf.add_to_collection('last_conv', net)

  with tf.name_scope('hyper_columns'):
    if is_training:
      # sample pixels corresponding to the last feature elements
      h, w = net.get_shape().as_list()[1:3]
      trace_locations = ops.trace_locations_backward
    else:
      # sample pixels corresponding to the whole image
      h, w = [height, width]
      trace_locations = ops.trace_locations_forward

    X, Y = tf.meshgrid(tf.range(w), tf.range(h), indexing='xy')
    loc_x = tf.tile(tf.reshape(X, [1,-1]), [num_batch, 1])
    loc_y = tf.tile(tf.reshape(Y, [1,-1]), [num_batch, 1])

    locations = [trace_locations(loc_x, loc_y, [h, w], [tf.shape(feat)[1], tf.shape(feat)[2]]) 
        for feat in hyperfeats]
    net = ops.extract_values(hyperfeats, locations)
    hyperchannels = net.get_shape().as_list()[-1]

    net = tf.reshape(net, [num_batch, h, w, hyperchannels]) 
    tf.add_to_collection('hyper_column', net)

    return net

def pixelnet(inputs, num_class, is_training=True, reuse=False, batch_norm=True):
  with nets.base_arg_scope(is_training, batch_norm):
    net = pixelnet_convs(inputs, num_class, is_training, reuse)
    with tf.variable_scope('aerial_mlp', reuse=reuse):
      fc1 = slim.conv2d(net, 512, [1,1], scope='fc1')
      fc2 = slim.conv2d(fc1, 512, [1,1], scope='fc2')
      fc3 = slim.conv2d(fc2, num_class, [1,1], scope='fc3',
          activation_fn=None, normalizer_fn=None)

      tf.add_to_collection('aerial_mlp', fc1)
      tf.add_to_collection('aerial_mlp', fc2)
      tf.add_to_collection('aerial_mlp', fc3)

  return fc3

def compute_indexing(source_size, target_size):
  # source_size is the size of reference feature map, where (0,0) 
  # corresponds to the top-left corner and (1,1) corresponds to the 
  # bottom-right conner of the feature map.
  
  jj, ii = np.meshgrid(range(source_size[1]), range(source_size[0]), indexing='xy')
  xx, yy = np.meshgrid(range(target_size[1]), range(target_size[0]), indexing='xy')
  X, I = np.meshgrid(xx.flatten(), ii.flatten(), indexing='xy')
  Y, J = np.meshgrid(yy.flatten(), jj.flatten(), indexing='xy')

  # normalize to 0 and 1
  I = I.astype('float32') / (source_size[0]-1)
  J = J.astype('float32') / (source_size[1]-1)
  Y = Y.astype('float32') / (target_size[0]-1)
  X = X.astype('float32') / (target_size[1]-1)

  indexing = tf.stack([I, J, Y, X], axis=2)
   
  return tf.expand_dims(indexing, 0)


def compute_transfweights(source_size, target_size, conditioned, 
    is_training=True, batch_norm=True, reuse=False):
  last_conv = tf.get_collection('last_conv')[0]
  batch_sz = last_conv.get_shape().as_list()[0]

  indexing_tensor = compute_indexing(source_size, target_size)
  indexing_tensor = tf.tile(indexing_tensor, [batch_sz, 1, 1, 1])
  H, W = indexing_tensor.get_shape().as_list()[1:3]

  if conditioned:
    # compute image global features
    with nets.base_arg_scope(is_training, batch_norm):
      with tf.variable_scope('condition_net', reuse=reuse) as scope:
        fc1 = slim.conv2d(last_conv, 64, [1,1], scope='fc1')
        tf.summary.histogram('fc1', fc1)
        fc2 = slim.conv2d(fc1, 1, [1,1], scope='fc2',
            activation_fn=None, normalizer_fn=None)
        tf.summary.histogram('fc2', fc2)
        fc2 = tf.reshape(fc2, [batch_sz, 1, 1, -1])
        fc2 = tf.tile(fc2, [1, H, W, 1])
        fc2 = tf.reshape(fc2, [batch_sz, H, W, -1])
    # concatenate with indexing tensor
    net = tf.concat([indexing_tensor, fc2], 3)
  else:
    net = indexing_tensor

  with nets.base_arg_scope(is_training, batch_norm, weight_decay=0.):
    with tf.variable_scope('weight_net', reuse=reuse) as scope:
      net = slim.conv2d(net, 128, [1,1], scope='fc1')
      tf.summary.histogram('fc1_wn', net)
      net = slim.conv2d(net, 64, [1,1], scope='fc2')
      tf.summary.histogram('fc2_wn', net)
      net = slim.conv2d(net, 1, [1,1], scope='fc3',
          activation_fn=None, normalizer_fn=None)
      tf.summary.histogram('fc3_wn', net)

  weights = tf.reshape(net, [-1, source_size[0]*source_size[1], target_size[0]*target_size[1]])
  weights = tf.nn.softmax(weights, dim=1) # sum columns to 1

  return weights

def transfnet(inputs, weights, target_size):
  # weights: batch x H_a*W_a x H_g*W_g
  b,h,w,c = inputs.get_shape().as_list()
  h_,w_ = target_size

  with tf.variable_scope('transfer_features') as scope:
    input_b_c_hw = tf.reshape(tf.transpose(inputs, [0,3,1,2]), [-1, c, h*w])
    output_b_c_hw = tf.matmul(input_b_c_hw, weights)
    output_b_c_h_w = tf.reshape(output_b_c_hw, [-1,c,h_,w_])
    output_b_h_w_c = tf.transpose(output_b_c_h_w, [0,2,3,1])

    biases = ops.constant_variable([1,h_,1,c], name='biases')
    output = output_b_h_w_c + biases

    tf.add_to_collection('transformer_weights', weights)
    tf.add_to_collection('transformer_weights', biases)
  return output
