# Learning Temporal Pose Estimation from Sparsely Labeled Videos (NeurIPS 2019)

## Introduction
This is an official pytorch implementation of [*Learning Temporal Pose Estimation from Sparsely Labeled Videos*](https://arxiv.org/abs/1906.04016). 
In this work, we introduce a framework that reduces the need for densely labeled video data, while producing strong pose detection performance. Our approach is useful even when training videos are densely labeled, which we demonstrate by obtaining state-of-the-art pose detection results on PoseTrack17 and PoseTrack18 datasets. Our method, called PoseWarper, is currently ranked first for multi-frame person pose estimation on [*PoseTrack leaderboard*](https://posetrack.net/leaderboard.php).

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

## Quick start
### Installation
1. Create a conda virtual environment and activate it.
   ```
   conda create -n posewarper python=3.7 -y
   source activate posewarper
   ```
2. Install pytorch v1.1.0.
   ```
   conda install pytorch=1.1.0 torchvision -c pytorch
   ```
3. Install mmcv.
   ```
   pip install mmcv
   ```
4. Install other dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   python setup.py install --user
   ```
6. Clone this repo. Let's refer to it as ${POSEWARPER_ROOT}.
7. Compile external modules:
   ```
   cd ${POSEWARPER_ROOT}/lib
   make
   cd ${POSEWARPER_ROOT}/lib/deform_conv
   python setup.py develop
   ```
8. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
    ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |   |-- hrnet_w48-8ef0771d.pth
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            |   |-- pose_resnet_101_256x192.pth
            |   |-- pose_resnet_101_384x288.pth
            |   |-- pose_resnet_152_256x192.pth
            |   |-- pose_resnet_152_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth
            `-- pose_mpii
                |-- pose_hrnet_w32_256x256.pth
                |-- pose_hrnet_w48_256x256.pth
                |-- pose_resnet_101_256x256.pth
                |-- pose_resnet_152_256x256.pth
                `-- pose_resnet_50_256x256.pth

   ```
  
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```


### Citation
If you use our code or models in your research, please cite our NeurIPS 2019 paper:
```
@inproceedings{NIPS2019_gberta,
title = {Learning Temporal Pose Estimation from Sparsely Labeled Videos},
author = {Bertasius, Gedas and Feichtenhofer, Christoph, and Tran, Du and Shi, Jianbo, and Torresani, Lorenzo},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2019},
}`

### Acknowledgement

Our PoseWarper implementation is built on top of [*Deep High Resolution Network implementation*](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). We thank the authors for releasing their code.

