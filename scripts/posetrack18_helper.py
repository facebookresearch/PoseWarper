# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
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

import os
import sys

#### environment variables
cur_python = '/path/to/your/python/binary'
working_dir = '/path/to/PoseWarper/'

### supplementary files
root_dir = '/path/to/our/provided/supplementary/files/directory/'

### directory with extracted and renamed frames
img_dir = '/path/to/posetrack18/'

### print frequency
PF = 5000

### Output Directories
baseline_output_dir = root_dir + 'posetrack18_experiments/baseline/'
if not os.path.exists(baseline_output_dir):
    os.makedirs(baseline_output_dir)

pose_warper_output_dir = root_dir + 'posetrack18_experiments/PoseWarper/'
if not os.path.exists(pose_warper_output_dir):
    os.makedirs(pose_warper_output_dir)

#### Paths to other files
json_dir = root_dir + 'posetrack18_json_files/'
pretrained_coco_model = root_dir + 'pretrained_models/pretrained_coco_model.pth'
precomputed_boxes_file = root_dir + 'posetrack18_precomputed_boxes/val_boxes.json'
annot_dir = root_dir + 'posetrack18_annotation_dirs/val/'

###########
for jj in range(2): ## train / inference
  V_list = [-1]
  N_list = [-1]
  for V in V_list: ## number of labeled videos
    for N in N_list: ## number of labeled frames per video
        if V < 0:
          N_str = '115'
          V_str = '250'
        else:
          N_str = str(N)
          V_str = str(V)

        #### Baseline
        cur_output_dir = baseline_output_dir + 'V'+V_str+'_N'+N_str + '/'
        if not os.path.exists(cur_output_dir):
            os.makedirs(cur_output_dir)

        out_dir = cur_output_dir + 'out/'
        log_dir = cur_output_dir + 'log/'

        #### training
        if jj == 0:
           epoch_sfx = ' TRAIN.END_EPOCH 10'
           batch_sfx = ''

           sfx = 'posetrack/pose_hrnet/w48_384x288_adam_lr1e-4/final_state.pth'
           inference_model_path = out_dir + sfx

           if not os.path.exists(inference_model_path):
             command = cur_python+' '+working_dir+'tools/train.py --cfg '+working_dir+'experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir +' MODEL.PRETRAINED ' +pretrained_coco_model+ ' PRINT_FREQ '+str(PF)+epoch_sfx+batch_sfx+' POSETRACK_ANNOT_DIR '+annot_dir+ ' DATASET.IS_POSETRACK18 True MODEL.EVALUATE False'
             #print(command)
             #print(xy)
             os.system(command)

        ##### inference
        if jj == 1:
           experiment_name = '"Baseline (# of Labeled Videos = '+V_str + '; # of Labeled Frames Per Video = '+N_str+')"'
           command = cur_python +' '+working_dir+'tools/test.py --cfg '+working_dir+'experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +inference_model_path+ ' TEST.COCO_BBOX_FILE '+precomputed_boxes_file+' POSETRACK_ANNOT_DIR '+annot_dir +' TEST.USE_GT_BBOX False PRINT_FREQ '+str(PF)+' EXPERIMENT_NAME ' +experiment_name+ ' DATASET.IS_POSETRACK18 True TEST.IMAGE_THRE 0.2'
           os.system(command)
           #print(command)
           #print(xy)

        sfx = 'posetrack/pose_hrnet/w48_384x288_adam_lr1e-4/final_state.pth'
        inference_model_path = out_dir + sfx

        #### PoseWarper
        cur_output_dir = pose_warper_output_dir + 'V'+V_str+'_N'+N_str+'/'
        if not os.path.exists(cur_output_dir):
            os.makedirs(cur_output_dir)

        out_dir = cur_output_dir + 'out/'
        log_dir = cur_output_dir + 'log/'
        preds_dir = out_dir

        #### training
        pretrained_model = inference_model_path
        if jj == 0:
           epoch_sfx = ' TRAIN.END_EPOCH 19'

           command = cur_python + ' '+working_dir+'tools/train.py --cfg '+working_dir+'experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_train.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir +' MODEL.PRETRAINED ' +pretrained_model+' POSETRACK_ANNOT_DIR '+annot_dir+ ' DATASET.IS_POSETRACK18 True MODEL.EVALUATE False PRINT_FREQ '+str(PF)
           #print(command)
           #print(xy)
           os.system(command)

        #### Temporal pose aggregation
        sfx = 'posetrack/pose_hrnet/w48_384x288_adam_lr1e-4_PoseWarper_train/final_state.pth'
        pose_warper_model_path = out_dir + sfx
        if jj == 1:
           experiment_name = '"Temporal Pose Aggregation via PoseWarper (# of Labeled Videos = '+V_str + '; # of Labeled Frames Per Video = '+N_str+')"'
           command = cur_python + ' '+working_dir+'tools/test.py --cfg '+working_dir+'experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_temporal_pose_aggregation.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +pose_warper_model_path+' TEST.COCO_BBOX_FILE '+precomputed_boxes_file+' POSETRACK_ANNOT_DIR '+annot_dir +' TEST.USE_GT_BBOX False EXPERIMENT_NAME '+experiment_name + ' DATASET.IS_POSETRACK18 True TEST.IMAGE_THRE 0.2 PRINT_FREQ '+str(PF)
           os.system(command)
           #print(command)
           #print(xy)
