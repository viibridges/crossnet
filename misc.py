import tensorflow as tf
import pprint, os
import numpy as np
from scipy.misc import imsave, imresize

pp = pprint.PrettyPrinter()
def pprint(obj):
  pp.pprint(obj)

def soft_labeling(im):
  im = im / (im.sum(-1)[...,None] + 1e-6)
  return im

def mkdir(dir_name):
  try:
    os.makedirs(dir_name)
    return dir_name
  except:
    return dir_name

def mkdir_for_file(file_name):
  dir_name = os.path.dirname(file_name)
  mkdir(dir_name)
  return file_name

def basename(full_path):
  return os.path.basename(full_path)

def preprocess_image(im, sz=None):
  """Preprocess image before training
  """
  im = tf.cast(im,tf.float32)/127.5 - 1. # to [-1,1]
  if sz:
    im = tf.image.resize_images(im, sz)    # resize

  return im

def preprocess_label(label, num_class, sz):
  """Make hard labels soft 
  """
  prob = tf.one_hot(label, num_class, axis=-1)
  prob = tf.image.resize_area(prob, sz)
  prob = tf.reshape(prob, [-1, sz[0], sz[1], num_class])
  return prob

def proprocess_image(im_, old_range=[-1.,1.], new_range=[0.,1.]):
  im = tf.identity(im_)
  im = tf.cast(im, tf.float32)
  im = (im - old_range[0]) / (old_range[1] - old_range[0]) # from old_range to [0,1]
  im = im * (new_range[1] - new_range[0]) + new_range[0]   # from [0,1] to new_range
  return im

def center_crop(im, sz):
  h, w = sz
  H, W = im.shape[:2]
  mx = int((W-w)/2)
  my = int((H-h)/2)
  return im[my:my+h, mx:mx+w, ...]

def to_montage(im_list, ncols=None, num_vis=6):
  if not ncols:
    batch = im_list[0].shape[0]
    ncols = min(batch, num_vis)
  nrows = len(im_list)

  max_dim = max(max([im.shape[1:3] for im in im_list]))
  h = w = min(max_dim, 256)
  image = np.zeros((h * nrows, w * ncols, 3))

  for i, im_batch in enumerate(im_list):
    for j, im in enumerate(im_batch[:ncols,...]):
      r = i*h; c = j*w
      if im.ndim == 2: # grayscale to color
        im = np.dstack([im]*3)
      if im.shape[-1] > 3:
        inspect_class_ids = [2,3,1] # (r,g,b) -> (road,tree,bldg)
        im = im[...,inspect_class_ids]
      image[r:r+h, c:c+w, :] = imresize(im, [h,w])

  return image

def pretty_transfmat(mat, source_size, target_size):
  hs, ws = source_size
  ht, wt = target_size
  mat_out = np.zeros([hs*ht, ws*wt])

  count = 0
  for i in xrange(ht):
    for j in xrange(wt):
      start_row = i*hs; end_row = start_row + hs
      start_col = j*ws; end_col = start_col + ws
      mat_out[start_row:end_row, start_col:end_col] = mat[:,count].reshape([hs, ws])
      count += 1

  return mat_out
