# Learning Temporal Pose Estimation from Sparsely Labeled Videos (NeurIPS 2019)

## Introduction
This is an official pytorch implementation of [*Learning Temporal Pose Estimation from Sparsely Labeled Videos*](https://arxiv.org/abs/1906.04016). 
In this work, we introduce a framework that reduces the need for densely labeled video data, while producing strong pose detection performance. Our approach is useful even when training videos are densely labeled, which we demonstrate by obtaining state-of-the-art pose detection results on PoseTrack17 and PoseTrack18 datasets. Our method, called PoseWarper, is currently ranked **first** for multi-frame person pose estimation on [*PoseTrack leaderboard*](https://posetrack.net/leaderboard.php).

## Results on the PoseTrack Dataset
### Temporal Pose Aggregation during Inference
| Method       |  Dataset Split | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean |
|--------------|--------------------|------|----------|-------|-------|------|------|-------|------|
| PoseWarper | val17  | 81.4 |     88.3 |  83.9 |  78.0 | 82.4 | 80.5 |  73.6 | 81.2 |
| PoseWarper | test17 | 79.5 |     84.3 |  80.1 |  75.8 | 77.6 | 76.8 |  70.8 | 77.9 |
| PoseWarper | val18  | 79.9 |     86.3 |  82.4 |  77.5 | 79.8 | 78.8 |  73.2 | 79.7 |
| PoseWarper | test18 | 78.9 |     84.4 |  80.9 |  76.8 | 75.6 | 77.5 |  71.8 | 78.0 |

### Video Pose Propagation on PoseTrack17
| Method                     | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean |
|--------------------|------|----------|-------|-------|------|------|-------|------|
| Pseudo-labeling w/HRNet   | 79.1 |     86.5 |  81.4 |  74.7 | 81.4 | 79.4 |  72.3 | 79.3 |
| FlowNet2 Propagation      | 82.7 |     91.0 |  83.8 |  78.4 | 89.7 | 83.6 |  78.1 | 83.8 |
| PoseWarper                | 86.0 |     92.7 |  89.5 |  86.0 | 91.5 | 89.1 |  86.6 | 88.7 |

## Environment
The code is developed using python 3.7, pytorch-1.1.0, and CUDA 10.0.1 on Ubuntu 18.04. For our experiments, we used 8 NVIDIA P100 GPUs.

## License
PoseWarper is released under the Apache 2.0 license.

## Quick start
### Installation
1. Create a conda virtual environment and activate it:
   ```
   conda create -n posewarper python=3.7 -y
   source activate posewarper
   ```
2. Install pytorch v1.1.0:
   ```
   conda install pytorch=1.1.0 torchvision -c pytorch
   ```
3. Install mmcv:
   ```
   pip install mmcv
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   python setup.py install --user
   ```
5. Clone this repo. Let's refer to it as ${POSEWARPER_ROOT}:
   ```
   git clone https://github.com/facebookresearch/PoseWarper.git
   ```
6. Install other dependencies:
   ```
   cd ${POSEWARPER_ROOT}
   pip install -r requirements.txt
   ```
7. Compile external modules:
   ```
   cd ${POSEWARPER_ROOT}/lib
   make
   cd ${POSEWARPER_ROOT}/lib/deform_conv
   python setup.py develop
   ```
8. Download our pretrained models, and some supplementary data files from [this link](https://www.dropbox.com/s/ygfy6r8nitoggfq/PoseWarper_supp_files.zip?dl=0) and extract it to ${POSEWARPER_SUPP_ROOT} directory.
  
### Data preparation
**For PoseTrack17 data**, we use a slightly modified version of the PoseTrack dataset where we rename the frames to follow `%08d` format, with first frame indexed as 1 (i.e. `00000001.jpg`). First, download the data from [PoseTrack download page](https://posetrack.net/users/download.php). Then, rename the frames for each video as described above using [this script](https://github.com/facebookresearch/DetectAndTrack/blob/master/tools/gen_posetrack_json.py). 

We provide all the required JSON files, which have already been converted to COCO format. Evaluation is performed using the official PoseTrack evaluation code, [poseval](https://github.com/leonid-pishchulin/poseval), which uses [py-motmetrics](https://github.com/cheind/py-motmetrics) internally. We also provide required MAT/JSON files that are required for the evaluation.

Your extracted PoseTrack17 images directory should look like this:
```
${POSETRACK17_IMG_DIR}
|-- bonn
`-- bonn_5sec
`-- bonn_mpii_test_5sec
`-- bonn_mpii_test_v2_5sec
`-- bonn_mpii_train_5sec
`-- bonn_mpii_train_v2_5sec
`-- mpii
`-- mpii_5sec
```

**For PoseTrack18 data**, please download the data from [PoseTrack download page](https://posetrack.net/users/download.php). Since the video frames are named properly, you only need to extract them into a directory of your choice (no need to rename the video frames). As with PoseTrack17, we provide all required JSON files for PoseTrack18 dataset as well.

Your extracted PoseTrack18 images directory should look like this:
```
${POSETRACK18_IMG_DIR}
|--images
`-- |-- test
    `-- train
    `-- val
```

### PoseTrack17 Experiments

First, you will need to modify [`scripts/posetrack17_helper.py`](scripts/posetrack17_helper.py) by setting appropriate path variables:
```
#### environment variables
cur_python = '/path/to/your/python/binary'
working_dir = '/path/to/PoseWarper/'

### supplementary files
root_dir = '/path/to/our/provided/supplementary/files/directory/'

### directory with extracted and renamed frames
img_dir = '/path/to/posetrack17/renamed_images/'
```

where working_dir=/path/to/PoseWarper/ should be the same as ${POSEWARPER_ROOT}, root_dir=/path/to/our/provided/supplementary/files/directory/ should be set to ${POSEWARPER_SUPP_ROOT}, and lastly img_dir=/path/to/posetrack17/renamed_images/ should point to ${POSETRACK17_IMG_DIR}.

After that, you can run the following PoseTrack17 experiments. All the output files, including the trained models will be saved in ${POSEWARPER_SUPP_ROOT}/posetrack17_experiments/ directory.

#### Video Pose Propagation
 
```
cd ${POSEWARPER_ROOT}
python scripts/posetrack17_helper.py 1
```

#### Data Augmentation with PoseWarper

```
cd ${POSEWARPER_ROOT}
python scripts/posetrack17_helper.py 2
```

#### Comparison to State-of-the-Art
```
cd ${POSEWARPER_ROOT}
python scripts/posetrack17_helper.py 3
```

#### All of the above experiments

```
cd ${POSEWARPER_ROOT}
python scripts/posetrack17_helper.py 0
```


### PoseTrack18 Experiments

First, you will need to modify [`scripts/posetrack18_helper.py`](scripts/posetrack18_helper.py) by setting appropriate path variables:
```
#### environment variables
cur_python = '/path/to/your/python/binary'
working_dir = '/path/to/PoseWarper/'

### supplementary files
root_dir = '/path/to/our/provided/supplementary/files/directory/'

### directory with extracted frames
img_dir = '/path/to/posetrack18/'
```

where working_dir=/path/to/PoseWarper/ should be the same as ${POSEWARPER_ROOT}, root_dir=/path/to/our/provided/supplementary/files/directory/ should be set to ${POSEWARPER_SUPP_ROOT}, and lastly img_dir=/path/to/posetrack18/ should point to ${POSETRACK18_IMG_DIR}.

After that, you can run the following PoseTrack18 experiment. All the output files, including the trained models will be saved in ${POSEWARPER_SUPP_ROOT}/posetrack18_experiments/ directory.

#### Comparison to State-of-the-Art

```
cd ${POSEWARPER_ROOT}
python scripts/posetrack18_helper.py
```
### Changing the Number of GPUs
Our experiments were conducted using 8 NVIDIA P100 GPUs. If you want to use a smaller number of GPUs, you need to modify *.yaml configuration files in [`experiments/posetrack/hrnet/`](experiments/posetrack/hrnet/). Specifically, you need to modify the GPUS entry in each configuration file. Depending on how many GPUs are used during training, you might also need to change TRAIN.BATCH_SIZE_PER_GPU entry in the configuration files. 

In addition to using 8 GPUs, we also tried using 4 GPUs for our experiments. Using a 4 GPU setup, we obtained similar results as with 8 GPUs without changing TRAIN.BATCH_SIZE_PER_GPU. However, note that the experiments will run substantially slower when smaller number of GPUs is used.

## Citation
If you use our code or models in your research, please cite our NeurIPS 2019 paper:
```
@inproceedings{NIPS2019_gberta,
title = {Learning Temporal Pose Estimation from Sparsely Labeled Videos},
author = {Bertasius, Gedas and Feichtenhofer, Christoph, and Tran, Du and Shi, Jianbo, and Torresani, Lorenzo},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2019},
}
```

## Acknowledgement

Our PoseWarper implementation is built on top of [*Deep High Resolution Network implementation*](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). We thank the authors for releasing their code.

