import os

import tensorflow as tf
import numpy as np
import preprocess_utils

def _preprocess_zero_mean_unit_range(inputs):
  """Map image values from [0, 255] to [-1, 1]."""
  return (2.0 / 255.0) * tf.to_float(inputs) - 1.0

def preprocess_image(image,
                     crop_height,
                     crop_width,
                     min_resize_value=None,
                     max_resize_value=None,
                     resize_factor=None,
                     min_scale_factor=1.,
                     max_scale_factor=1.,
                     scale_factor_step_size=0,
                     is_training=True):
  original_image = image
  processed_image = tf.cast(image, tf.float32)
  if (min_resize_value is not None or max_resize_value is not None):
    [processed_image] = \
        preprocess_utils.resize_to_range(
            image=processed_image,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True)
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  '''
  # Data augmentation by randomly scaling the inputs.
  scale = preprocess_utils.get_random_scale(
      min_scale_factor, max_scale_factor, scale_factor_step_size)
  processed_image = preprocess_utils.randomly_scale_image(
      processed_image, scale)
  processed_image.set_shape([None, None, 3])

  # Pad image with mean pixel value.
  if is_training:
    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0] # vis 508
    image_width = image_shape[1]
    mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])
    target_height = image_height + tf.maximum(crop_height - image_height, 0) # 448
    target_width = image_width + tf.maximum(crop_width - image_width, 0) # 256
    processed_image = preprocess_utils.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)
  '''
  # Randomly crop the image and label.
  if is_training:
    [processed_image] = preprocess_utils.random_crop(
        [processed_image], crop_height, crop_width)
  else:
    processed_image = tf.image.resize_image_with_crop_or_pad(processed_image, crop_height, crop_width)

  processed_image.set_shape([crop_height, crop_width, 3])

  if is_training:
    # Randomly left-right flip the image and label.
    processed_image, _ = preprocess_utils.flip_dim(
        [processed_image], 0.5, dim=1)

  processed_image = _preprocess_zero_mean_unit_range(processed_image)

  return processed_image


class ImageNetDataSet(object):
  def __init__(self, data_dir, subset='train', use_distortion=True):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion

  def get_filenames(self):
    if self.subset in ['train', 'validation', 'eval']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.image.decode_jpeg(features['image'], channels=3)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width, 3])


    label = tf.cast(features['label'], tf.int32)
    image = tf.cast(image, tf.float32)
    image = self.preprocess(image)

    return image, label

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
        return preprocess_image(image, 224,224, 256,None,None,0.25,1,0,True)
    else:
        return preprocess_image(image, 224,224, 256,None,None,1.0,1.0,0,False)

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse records.
    #dataset = dataset.map(
    #    self.parser, num_threads=batch_size, output_buffer_size=2 * batch_size)
    dataset = dataset.map(self.parser, num_parallel_calls=batch_size)
    dataset = dataset.repeat(None)
    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          ImageNetDataSet.num_examples_per_epoch(self.subset) * 0.04)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      #dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
      dataset = dataset.shuffle(buffer_size=20480, reshuffle_each_iteration=True)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 1281167
    elif subset == 'validation':
      return 50000
    elif subset == 'eval':
      return 50000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
