# MobileNetV2 SSDLite (Single Shot MultiBox Detector Lite) - Tensorflow 2.0

**Modify from [ssd-tf2](https://github.com/ChunML/ssd-tf2).**

## Preparation
- Download PASCAL VOC dataset (2007 or 2012) and extract at `./data`
- Install necessary dependencies:
```
pip install -r requirements.txt
```

## Training
Arguments for the training script:

```
>> python train.py --help
usage: train.py [-h] [--data-dir DATA_DIR] [--data-type DATE_TYPE]
                [--arch ARCH] [--batch-size BATCH_SIZE]
                [--num-batches NUM_BATCHES] [--neg-ratio NEG_RATIO]
                [--initial-lr INITIAL_LR] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--num-epochs NUM_EPOCHS]
                [--checkpoint-dir CHECKPOINT_DIR]
                [--pretrained-type PRETRAINED_TYPE] 
                [--pretrained-dir PRETRAINED_DIR][--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder for VOC)
-  `--data-type` the type of the dataset (VOC2007 or VOC2012 or CCPD or TFRecord)
-  `--arch` SSD network architecture (ssdlite)
-  `--batch-size` training batch size
-  `--num-batches` number of batches to train (`-1`: train all)
-  `--neg-ratio` ratio used in hard negative mining when computing loss
-  `--initial-lr` initial learning rate
-  `--momentum` momentum value for SGD
-  `--weight-decay` weight decay value for SGD
-  `--num-epochs` number of epochs to train
-  `--checkpoint-dir` checkpoint directory
-  `--pretrained-type` pretrained weight type (`base`: using pretrained MobileNetV2 backbone, other options: see testing section)
-  `--pretrained-dir` pretrained weight
-  `--gpu-id` GPU ID

- how to train SSDLite using PASCAL VOC2007 for 100 epochs:

```
python train.py --data-dir ./data/VOCdevkit --data-type 2007 --num-epochs 100
```

- how to train SSDLite using pretrained model for different number of class:
```
python train.py --dat-dir ./data/CCPD --data-type CCPD --pretrained-type transfer --pretrained-dir ./pretrained
```

## Testing
Arguments for the testing script:
```
>> python test.py --help
usage: test.py [-h] [--data-dir DATA_DIR] [--data-year DATA_YEAR]
               [--arch ARCH] [--num-examples NUM_EXAMPLES]
               [--pretrained-type PRETRAINED_TYPE]
               [--checkpoint-dir CHECKPOINT_DIR]
               [--checkpoint-path CHECKPOINT_PATH] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--data-year` the year of the dataset (2007 or 2012)
-  `--arch` SSD network architecture (ssdlite)
-  `--num-examples` number of examples to test (`-1`: test all)
-  `--checkpoint-dir` checkpoint directory
-  `--checkpoint-path` path to a specific checkpoint
-  `--pretrained-type` pretrained weight type (`latest`: automatically look for newest checkpoint in `checkpoint_dir`, `specified`: use the checkpoint specified in `checkpoint_path`, `transfer`: use the pretrained model to transfer learning)
-  `--gpu-id` GPU ID

- how to test the first training pattern above using the latest checkpoint:

```
python test.py --data-dir ./data/VOCdevkit --data-year 2007 --checkpoint_dir ./checkpoints
```

- how to test the second training pattern above using the 100th epoch's checkpoint, using only 40 examples:

```
python test.py --data-dir ./data/VOCdevkit --data-year 2012 --arch ssdlite --checkpoint_path ./checkpoints_512/ssdlite_epoch_100.h5 --num-examples 40
```

## Reference
- Single Shot Multibox Detector paper: [paper](https://arxiv.org/abs/1512.02325)
- Keras implementation: [code](https://github.com/markshih91/mobilenet_v2_ssdlite_keras)
- Pytorch implementation: [code](https://github.com/qfgaohao/pytorch-ssd)