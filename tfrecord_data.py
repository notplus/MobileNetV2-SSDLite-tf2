import tensorflow as tf
import os
import numpy as np

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# def _extract_fn(tfrecord):
def extract_fn(augmentation, default_boxes, tfrecord):
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, image_feature_description)

    filename = sample['image/filename']
    img = tf.io.decode_jpeg(sample['image/encoded'])
    height = sample['image/height']
    width = sample['image/width']
    
    xmin = sample['image/object/bbox/xmin']
    xmax = sample['image/object/bbox/xmax']
    ymin = sample['image/object/bbox/ymin']
    ymax = sample['image/object/bbox/ymax']


    boxes = [[xmin, ymin, xmax, ymax]]
    labels = [1]
    
    # boxes = tf.constant(boxes, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.int64)

    # augmentation_method = np.random.choice(augmentation)
    # if augmentation_method == 'patch':
    #     img, boxes, labels = random_patching(img, boxes, labels)
    # elif augmentation_method == 'flip':
    #     img, boxes, labels = horizontal_flip(img, boxes, labels)

    img = tf.cast(img, tf.float32)
    img = (img / 127.0) - 1.0
    
    gt_confs, gt_locs = compute_target(
        default_boxes, boxes, labels)

    return filename, img, gt_confs, gt_locs
    
