import tensorflow as tf
import os
from functools import partial
from voc_data import VOCDataset
from tfrecord_data import extract_fn
from ccpd_data import CCPDDataset

def create_batch_generator(root_dir, type, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):

    if type.startswith('VOC'):
        num_examples = batch_size * num_batches if num_batches > 0 else -1
        voc = VOCDataset(root_dir, type, default_boxes,
                        new_size, num_examples, augmentation)

        info = {
            'idx_to_name': voc.idx_to_name,
            'name_to_idx': voc.name_to_idx,
            'length': len(voc),
            'image_dir': voc.image_dir,
            'anno_dir': voc.anno_dir
        }

        if mode == 'train':
            train_gen = partial(voc.generate, subset='train')
            train_dataset = tf.data.Dataset.from_generator(
                train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
            val_gen = partial(voc.generate, subset='val')
            val_dataset = tf.data.Dataset.from_generator(
                val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

            train_dataset = train_dataset.shuffle(40).batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)

            return train_dataset.take(num_batches), val_dataset.take(-1), info
        else:
            dataset = tf.data.Dataset.from_generator(
                voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
            dataset = dataset.batch(batch_size)
            return dataset.take(num_batches), info
    elif type.startswith('TFRecord'):
        num_examples = batch_size * num_batches if num_batches > 0 else -1
    
        if augmentation == None:
            augmentation = ['original']
        else:
            augmentation = augmentation + ['original']

        if mode == 'train':
            raw_train_dataset = tf.data.TFRecordDataset(os.path.join(root_dir, 'train.tfrecord'))
            train_dataset = raw_train_dataset.map(partial(extract_fn, augmentation, default_boxes))

            raw_val_dataset = tf.data.TFRecordDataset(os.path.join(root_dir, 'val.tfrecord'))
            val_dataset = raw_val_dataset.map(partial(extract_fn, augmentation, default_boxes))

            train_dataset = train_dataset.shuffle(40).batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)

            train_dataset_len = train_dataset.reduce(0, lambda x,_: x+1).numpy() * batch_size
            val_dataset_len = val_dataset.reduce(0, lambda x,_: x+1).numpy() * batch_size

            info = {
                'length': train_dataset_len + val_dataset_len,
                'idx_to_name': ['plate'],
            }

            return train_dataset.take(num_batches), val_dataset.take(-1), info
        else:
            raw_test_dataset = tf.data.TFRecordDataset(os.join(root_dir, 'test.tfrecord'))
            test_dataset = raw_test_dataset.map(partial(root_dir, augmentation, default_boxes))

            test_dataset = test_dataset.take(num_examples).batch(batch_size)

            info = {
                'length': len(test_dataset),
                'idx_to_name': ['plate'],
            }

            return test_dataset.take(num_batches), info
    elif type.startswith('CCPD'):
        num_examples = batch_size * num_batches if num_batches > 0 else -1
        ccpd = CCPDDataset(root_dir, default_boxes,
                        new_size, num_examples, augmentation)

        info = {
            'length': len(ccpd),
            'image_dir': root_dir,
            'idx_to_name': ['plate'],
        }

        if mode == 'train':
            train_gen = partial(ccpd.generate, subset='train')
            train_dataset = tf.data.Dataset.from_generator(
                train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
            val_gen = partial(ccpd.generate, subset='val')
            val_dataset = tf.data.Dataset.from_generator(
                val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

            train_dataset = train_dataset.shuffle(40).batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)

            return train_dataset.take(num_batches), val_dataset.take(-1), info
        else:
            dataset = tf.data.Dataset.from_generator(
                ccpd.generate, (tf.string, tf.float32, tf.int64, tf.float32))
            dataset = dataset.batch(batch_size)
            return dataset.take(num_batches), info