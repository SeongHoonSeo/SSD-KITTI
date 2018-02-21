# Single Shot MultiBox Detector(SSD) on KITTI Dataset

## Regarding Copyright/License

**This is a forked repository from the [SDC-Vehicle-Detection](https://github.com/balancap/SDC-Vehicle-Detection) by [Paul Balanca](https://github.com/balancap).**  
**All copyrights are as mentioned in the source code, and retained to the rightful owners.**  

Though there isn't any `license.txt` or `license.md` file within the original repository,
copyrights **are mentioned** at the head of each file.
Parts of the source code that explain copyright/license are exactly identical to that of the original repository.  
**They are left intact.**  

* Example of a Copyright header within a source code
```
#Copyright 2016 Paul Balanca. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Goal of this Repo

**The Goal of this repository is to enhance the performance of SSD for [KITTI 2D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)**  

Original implementation of SSD has square-sized input layer(300x300, 512x512).  
However, the resolution of KITTI dataset images is 375x1242, with width being more than 3 times longer than the height.  
Due to this fact, simple resizing of the image to 300x300 often leads to poor detection accuracy.  

We will tackle this problem by modifying several parts of the SSD (ex. resolution of Conv layers, filters, preprocessing methods, etc.)

## Train and Validate the Model

### 0. Prerequisites

1. NVIDIA graphic driver (if planning to use GPU-version of tensorflow)
2. CUDA Toolkit
3. CuDNN
4. Anaconda Python 3.6
5. Bazel
6. Tensorflow

If you can read Korean and are using ubuntu, [here](http://ejklike.github.io/2017/03/06/install-tensorflow1.0-on-ubuntu16.04-1.html) is a website that well guides through the entire installation process.

### 1. Convert KITTI Dataset to tfrecord file

1. Download KITTI 2D Object Detection Dataset from the [official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
    * left color images of object data set (12GB)
    * training labels of object data set (5MB)
    * (Optional) Object devlopment kit (1MB) if you want to know more about KITTI Benchmark Suite

2. Convert KITTI Dataset to tfrecord file

`dataset_dir`: directory where KITTI dataset is located  
`output_name`: kitti_train or kitti_test  
`output_dir`: directory where tfrecord file will be located  
`need_validation`: True (split training dataset to train and validation) or False (do not split)  

* Training Dataset
```
python3 tf_convert_data.py \
    --dataset_name=kitti \
    --dataset_dir=/data/KITTI/training \ 
    --output_name=kitti_train \ 
    --output_dir=/data/KITTI/training \ 
    --need_validation_split=True \ 
```

* Test Dataset
```
python3 tf_convert_data.py \
    --dataset_name=kitti \
    --dataset_dir=/data/KITTI/test \
    --output_name=kitti_test \ 
    --output_dir=/data/KITTI/test \
    --need_validation_split=False \
```

### 2. Choose which model to train

There are several variations of the SSD model in this repository
* `ssd_vgg_300`: original SSD (300x300) implementation 
* `ssd_vgg_384x1280`: SSD with enlarged input resolution as 384x1280. 
* `ssd_vgg_384x1280_AnchorIncrease`: 384x1280 input resolution with bigger anchor sizes.
* `ssd_vgg_384x1280_modified`: 384x1280 input resolution with increased anchor sizes (and modified caulcation method of x, y coordinates of anchor boxes)
* `ssd_vgg_384x640`: SSD with enlarged input resolution as 384x640.  

(Filter shapes are identical to that of the original SSD, so pretrained weights can be applied to all variations)

To add your own model of SSD,
1. modify `preprocessing_factory.py`
2. modify `nets_factory.py`

### 3. Train the SSD model

`train_dir`: directory where the trained model will be saved  
`dataset_dir`: directory where the kitti_train.tfrecord (or kitti_full_train.tfrecord) file is located  
`checkpoint_path`: directory where the pretrained model is located (ssd_model.ckpt is the model pretrained on PASAL VOC dataset)  
`checkpoint_exclude_scopes`: filters where pretrained weights are NOT applied  
`dataset_split_name`: train or full_train (full_train consists of all 7481 training images without validation split)  
`model_name`: name of the SSD model (choose from 2. or select your own!)

```
python3 train_ssd_network.py \
    --train_dir=./logs/ \
    --dataset_dir=/data/KITTI/training \
    --checkpoint_path=./checkpoints/ssd_model.ckpt \
    --checkpoint_exclude_scopes=ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --dataset_name=kitti \
    --dataset_split_name=train \
    --model_name=ssd_vgg_300 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.0005 \
    --optimizer=rmsprop \
    --learning_rate=0.001 \
    --batch_size=4
```

### 4. Validate the SSD model

`eval_dir`: directory where the evaluation results will be saved  
`dataset_dir`: directory wehre kitti_validation.tfrecord file is located  
`dataset_split_name`: validation  
`model_name`: name of the SSD model (choose from 2. or select your own!)  
`checkpoint_path`: directory where the trained model is located  

```
python3 eval_ssd_network.py \
    --eval_dir=./eval/SSD300 \
    --dataset_dir=/home/intern/data/KITTI/training \
    --dataset_split_name=validation \
    --model_name=ssd_vgg_300 \
    --checkpoint_path=./models/ssd_300_kitti \
    --batch_size=1 \
```

(Note that we cannot test SSD model on KITTI test dataset, because KITTI benchmark suite does not provide labels for test images.)


## Evaluation Results

(values are rounded rounded to the fourth decimal place)  

**1. Mean Average Precision - PASCAL VOC07 Method**  

|                     |                    | 300x300 | 384x640 | 384x1280 | 384x1280_modified |
|:-------------------:|:------------------:|:-------:|:-------:|:--------:|:-----------------:|
|       AP_VOC07      |       Car (1)      |  0.5961 |  0.7421 |  0.6942  |       0.6851      |
|                     |       Van (2)      |  0.3931 |  0.4359 |  0.5779  |       0.6214      |
|                     |      Truck (3)     |  0.2985 |  0.5659 |  0.5496  |       0.6195      |
|                     |     Cyclist (4)    |  0.1066 |  0.1581 |  0.4179  |       0.2480      |
|                     |   Pedestrian (5)   |  0.0909 |  0.1432 |  0.3709  |       0.1982      |
|                     | Person_sitting (6) |    0    |  0.0207 |  0.4881  |       0.2674      |
|                     |      Tram (7)      |  0.2360 |  0.4231 |  0.3410  |       0.7030      |
|                     |      Misc (8)      |  0.1995 |  0.2911 |  0.3687  |       0.5262      |
|      mAP_VOC07      |                    |  0.2401 |  0.3475 |  0.4760  |       0.4835      |
| mAP_VOC07 (1, 4, 5) |                    |  0.2645 |  0.3478 |  0.4943  |       0.3771      |


**2. Mean Average Precision - PASCAL VOC12 Method**  

|                     |                    | 300x300 | 384x640 | 384x1280 | 384x1280_modified |
|:-------------------:|:------------------:|:-------:|:-------:|:--------:|:-----------------:|
|       AP_VOC12      |       Car (1)      |  0.6049 |  0.7778 |  0.7174  |       0.7057      |
|                     |       Van (2)      |  0.3799 |  0.4230 |  0.5929  |       0.6292      |
|                     |      Truck (3)     |  0.2764 |  0.5658 |  0.5614  |       0.6351      |
|                     |     Cyclist (4)    |  0.0375 |  0.1404 |  0.4179  |       0.2211      |
|                     |   Pedestrian (5)   |  0.0147 |  0.0903 |  0.3536  |       0.1411      |
|                     | Person_sitting (6) |    0    |  0.0156 |  0.4864  |       0.2524      |
|                     |      Tram (7)      |  0.2230 |  0.4118 |  0.3116  |       0.7329      |
|                     |      Misc (8)      |  0.1689 |  0.2677 |  0.3645  |       0.5273      |
|      mAP_VOC12      |                    |  0.2132 |  0.3365 |  0.4757  |       0.4806      |
| mAP_VOC12 (1, 4, 5) |                    |  0.2190 |  0.3362 |  0.4963  |       0.3560      |

**3. Evaluation Time**  
* Time to evaluate mAP in both VOC07 and VOC12 methods
* Size of validation set: 749 (10% of the training set)
* Batch Size = 1  

|                   | 300x300 | 384x640 | 384x1280 | 384x1280_modified |
|:-----------------:|:-------:|:-------:|:--------:|:-----------------:|
| Total Time (sec.) |  93.111 | 159.342 |  269.900 |      265.913      |
|  Per Batch (sec.) |  0.124  |  0.213  |   0.360  |       0.355       |



**4. Observations**  
1. Simple warp-resize of rectangular image to square images lead to considerable loss in performance, especially in detecting "thin" objects such as pedestrian and cyclist.
In general, preserving the original resolution results in higher detection accuracy.

2. Increasing the size of anchors leads to higher detection accuracy for big objects (ex. Van, Truck, Tram), but leads to lower accuracy regarding smaller objects (ex. Car, Pedestrian)  

3. Detection accuracy for car is highest when the input resolution is 384x640. It seems that higher input resolution leads to unintentional increase in width and height of anchors. From this, we could speculate that many cars in the image are "far-away", having small bounding boxes.  
(Code from `ssd_vgg_300.py`)
```
    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w
```

## Milestones

2018.02.12: [Fixed syntax errors and enabled training](https://github.com/seonghoon247/SSD-KITTI/commit/b080149b98e07f80d982a66137bcee374d2dd8a2)  
2018.02.13: [Update outdated code by referencing balancap/tensorflow repo](https://github.com/seonghoon247/SSD-KITTI/commit/222194ee0241bc5f8fa00a331cf52742084df554)  
2018.02.14: [Implemented evaluation code for KITTI](https://github.com/seonghoon247/SSD-KITTI/commit/c7ae332b036ca5ef9225b6b6c7befdf0e8921273)  
2018.02.19: [Fixed num_class to 9 and removed don't-care labels from training](https://github.com/seonghoon247/SSD-KITTI/commit/fc6efe53d4d9b6259b92100ada85ca6d59f8e525)  
2018.02.20: [Implemented train/validation split](https://github.com/seonghoon247/SSD-KITTI/commit/79f53d6f45a5480bbc7d2b72a04fdd909c71ffee)  
2018.02.21: [Modify the codes to enable train/test on various models](https://github.com/seonghoon247/SSD-KITTI/commit/5ac0124316aa6a33c7972082b73cd796fbf48c24)  

## Future Works
1. Further Understanding and Refactoring of the Codes (Some outdated codes are still mixed)

2. Find out why training losses are high despite of seemingly "correct" detection accuracy

3. Further modify SSD network, including modification of filter sizes or 'feature pyramids'

4. Further Explore tradeoff between detection time and accuracy

## To end...

This project is done by [Seong Hoon Seo](https://github.com/seonghoon247) and [GeonSeok Seo](https://github.com/geonseoks) with grateful advice and help from JisooJeong