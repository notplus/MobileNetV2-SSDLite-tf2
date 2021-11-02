from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os


from layers import create_mobilenetv2_layers, predict_blocks


def create_ssdlite_model(image_size,
                         num_classes,
                         aspect_ratios_per_layer=[[1.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                  [1.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                  [1.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                  [1.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                  [1.0, 1.0, 2.0, 0.5],
                                                  [1.0, 1.0, 2.0, 0.5]]):

    # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_predictor_layers = 6

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, \
                but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    n_boxes = []
    for ar in aspect_ratios_per_layer:
        n_boxes.append(len(ar))

    x = layers.Input(shape=(img_height, img_width, img_channels))

    links = create_mobilenetv2_layers(x)

    link1_cls = predict_blocks(links[0], n_boxes[0] * num_classes, 'cls', 1)
    link2_cls = predict_blocks(links[1], n_boxes[1] * num_classes, 'cls', 2)
    link3_cls = predict_blocks(links[2], n_boxes[2] * num_classes, 'cls', 3)
    link4_cls = predict_blocks(links[3], n_boxes[3] * num_classes, 'cls', 4)
    link5_cls = predict_blocks(links[4], n_boxes[4] * num_classes, 'cls', 5)
    link6_cls = predict_blocks(links[5], n_boxes[5] * num_classes, 'cls', 6)
    
    link1_box = predict_blocks(links[0], n_boxes[0] * 4, 'box', 1)
    link2_box = predict_blocks(links[1], n_boxes[1] * 4, 'box', 2)
    link3_box = predict_blocks(links[2], n_boxes[2] * 4, 'box', 3)
    link4_box = predict_blocks(links[3], n_boxes[3] * 4, 'box', 4)
    link5_box = predict_blocks(links[4], n_boxes[4] * 4, 'box', 5)
    link6_box = predict_blocks(links[5], n_boxes[5] * 4, 'box', 6)

    # Reshape
    # cls1_reshape = layers.Reshape((-1, num_classes), name='ssd_cls1_reshape')(link1_cls)
    # cls2_reshape = layers.Reshape((-1, num_classes), name='ssd_cls2_reshape')(link2_cls)
    # cls3_reshape = layers.Reshape((-1, num_classes), name='ssd_cls3_reshape')(link3_cls)
    # cls4_reshape = layers.Reshape((-1, num_classes), name='ssd_cls4_reshape')(link4_cls)
    # cls5_reshape = layers.Reshape((-1, num_classes), name='ssd_cls5_reshape')(link5_cls)
    # cls6_reshape = layers.Reshape((-1, num_classes), name='ssd_cls6_reshape')(link6_cls)

    # box1_reshape = layers.Reshape((-1, 4), name='ssd_box1_reshape')(link1_box)
    # box2_reshape = layers.Reshape((-1, 4), name='ssd_box2_reshape')(link2_box)
    # box3_reshape = layers.Reshape((-1, 4), name='ssd_box3_reshape')(link3_box)
    # box4_reshape = layers.Reshape((-1, 4), name='ssd_box4_reshape')(link4_box)
    # box5_reshape = layers.Reshape((-1, 4), name='ssd_box5_reshape')(link5_box)
    # box6_reshape = layers.Reshape((-1, 4), name='ssd_box6_reshape')(link6_box)
    
    cls1_reshape = tf.reshape(link1_cls, [-1, link1_cls.shape[1]*link1_cls.shape[2]*n_boxes[0], num_classes])
    cls2_reshape = tf.reshape(link2_cls, [-1, link2_cls.shape[1]*link2_cls.shape[2]*n_boxes[1], num_classes])
    cls3_reshape = tf.reshape(link3_cls, [-1, link3_cls.shape[1]*link3_cls.shape[2]*n_boxes[2], num_classes])
    cls4_reshape = tf.reshape(link4_cls, [-1, link4_cls.shape[1]*link4_cls.shape[2]*n_boxes[3], num_classes])
    cls5_reshape = tf.reshape(link5_cls, [-1, link5_cls.shape[1]*link5_cls.shape[2]*n_boxes[4], num_classes])
    cls6_reshape = tf.reshape(link6_cls, [-1, link6_cls.shape[1]*link6_cls.shape[2]*n_boxes[5], num_classes])

    box1_reshape = tf.reshape(link1_box, [-1, link1_box.shape[1]*link1_box.shape[2]*n_boxes[0], 4])
    box2_reshape = tf.reshape(link2_box, [-1, link2_box.shape[1]*link2_box.shape[2]*n_boxes[1], 4])
    box3_reshape = tf.reshape(link3_box, [-1, link3_box.shape[1]*link3_box.shape[2]*n_boxes[2], 4])
    box4_reshape = tf.reshape(link4_box, [-1, link4_box.shape[1]*link4_box.shape[2]*n_boxes[3], 4])
    box5_reshape = tf.reshape(link5_box, [-1, link5_box.shape[1]*link5_box.shape[2]*n_boxes[4], 4])
    box6_reshape = tf.reshape(link6_box, [-1, link6_box.shape[1]*link6_box.shape[2]*n_boxes[5], 4])

    # cls = layers.Concatenate(axis=1, name='ssd_cls')(
    #     [cls1_reshape, cls2_reshape, cls3_reshape, cls4_reshape, cls5_reshape, cls6_reshape]
    # )

    # box = layers.Concatenate(axis=1, name='ssd_box')(
    #     [box1_reshape, box2_reshape, box3_reshape, box4_reshape, box5_reshape, box6_reshape]
    # )

    cls = tf.concat([cls1_reshape, cls2_reshape, cls3_reshape, cls4_reshape, cls5_reshape, cls6_reshape], axis=1)
    box = tf.concat([box1_reshape, box2_reshape, box3_reshape, box4_reshape, box5_reshape, box6_reshape], axis=1)

    # predictions = layers.Concatenate(axis=2, name='ssd_predictions')([cls, box])

    model = Model(inputs=x, outputs=[cls,box])

    return model

class SSDLite(Model):
    """ Class for SSDLite model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch='ssd300'):
        super(SSDLite, self).__init__()
        self.num_classes = num_classes
        self.links = create_mobilenetv2_layers()


    def compute_heads(self, x, idx):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        conf = self.conf_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        loc = self.loc_head_layers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    # TODO: initialize mobilenetv2 layers from pretrained weights
    def init_mobilnetv2(self):
        origin_mbv2 = MobileNetV2()

    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0

        return confs, locs

def init_mobilnetv2(net):
    origin_mbv2 = MobileNetV2()
    j = 1
    for i in range(1, 150):
        if origin_mbv2.get_layer(index=j).name.endswith('pad'):
            j += 1
        net.get_layer(index=i).set_weights(origin_mbv2.get_layer(index=j).get_weights())
        j += 1
    return net

def create_ssdlite(num_classes, arch, pretrained_type,
               checkpoint_dir=None,
               checkpoint_path=None,
               pretrained_dir=None):
    """ Create SSDLite model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    net = create_ssdlite_model([300, 300, 3],num_classes)

    # net(tf.random.normal((1, 512, 512, 3)))
    if pretrained_type == 'base':
        net = init_mobilnetv2(net)
    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                     for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            print(latest)
            net.load_weights(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            # net.init_vgg16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')
    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            print(e)
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))
    elif pretrained_type == 'transfer':
        try:
            ssdlite = tf.keras.models.load_model(pretrained_dir)
            for i in range(1, 222):
                net.get_layer(index=i).set_weights(ssdlite.get_layer(index=i).get_weights())
            
        except Exception as e:
            print(e)
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    
    return net
