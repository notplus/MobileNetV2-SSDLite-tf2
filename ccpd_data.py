import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random

from tensorflow.keras import layers
import yaml
from anchor import generate_default_boxes

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial


class CCPDDataset():
    """ Class for CCPD Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/ccpd_base)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, root_dir, default_boxes,
                 new_size, num_examples=-1, augmentation=None):
        super(CCPDDataset, self).__init__()
        self.data_dir = os.path.join(root_dir)
        self.ids = list(map(lambda x: x[:-4], os.listdir(self.data_dir)))
        self.default_boxes = default_boxes
        self.new_size = new_size

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 0.9)]
        self.val_ids = self.ids[int(len(self.ids) * 0.9):]

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file and crop size to (720, 720)
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 720, 720)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.data_dir, filename + '.jpg')
        img = Image.open(img_path).crop((0, 0, 720, 720))

        return img

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        filename = self.ids[index]

        kp = []
        kp_serial = filename.split('-')[3].split('_')

        for i in range(0, 4):
            tmp = kp_serial[i].split('&')
            kp.append(int(tmp[0]))
            kp.append(int(tmp[1]))

        # top left point
        tlx = min(kp[2], kp[4]) / w
        tly = min(kp[5], kp[7]) / h

        # bottom right point
        brx = max(kp[0], kp[6]) / w
        bry = max(kp[1], kp[3]) / h

        box = [[tlx, tly, brx, bry]]
        label = [1]

        return np.array(box, dtype=np.float32), np.array(label,dtype=np.int64)


    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'patch':
                img, boxes, labels = random_patching(img, boxes, labels)
            elif augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)

            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)
            
            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels)

            yield filename, img, gt_confs, gt_locs
