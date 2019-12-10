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

### experiment selection
experiment = int(sys.argv[1])

#### environment variables
cur_python = '/path/to/your/python/binary'
working_dir = '/path/to/PoseWarper/'

### supplementary files
root_dir = '/path/to/our/provided/supplementary/files/directory/'

### directory with extracted and renamed frames
img_dir = '/path/to/posetrack17/renamed_images/'

if experiment == 0: ### run all experiments
    rs = 0
    rf = 4
elif experiment == 1: ## video pose propagation experiment
    rs = 0
    rf = 1
elif experiment == 2: ## Data Augmentation via PoseWarper and Temporal Pose Aggregation during inference
    rs = 1
    rf = 3
elif experiment == 3: ## state-of-the-art expeirments using full PoseTrack17 data
    rs = 3
    rf = 4
#################

### print frequency
PF = 5000

#### Paths to other files
json_dir = root_dir + 'posetrack17_json_files/'
pretrained_coco_model = root_dir + 'pretrained_models/pretrained_coco_model.pth'
precomputed_boxes_file = root_dir + 'posetrack17_precomputed_boxes/val_boxes.json'
annot_dir = root_dir + 'posetrack17_annotation_dirs/'

### Output Directories
baseline_output_dir = root_dir + 'posetrack17_experiments/baseline/'
if not os.path.exists(baseline_output_dir):
    os.makedirs(baseline_output_dir)

pose_warper_output_dir = root_dir + 'posetrack17_experiments/PoseWarper/'
if not os.path.exists(pose_warper_output_dir):
    os.makedirs(pose_warper_output_dir)

data_aug_output_dir = root_dir + 'posetrack17_experiments/baseline_wPoseWarperDataAug/'
if not os.path.exists(data_aug_output_dir):
    os.makedirs(data_aug_output_dir)

###########
for jj in range(2): ## train / inference
 for rr in range(rs,rf): ## different modes
  if rr == 0: ## video pose propagation
    V_list = [250]
    N_list = [5]
  elif rr == 1: ### data augmentation via posewarper and Temporal Pose Aggregation during Inference
    V_list = [100, 175, 250]
    N_list = [1]
  elif rr == 2: ### data augmentation via posewarper and Temporal Pose Aggregation during Inference
    V_list = [50]
    N_list = [1, 2, 4]
  else: ## state-of-the-art experiments
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
           if rr != 3:
             epoch_sfx = ''
             batch_sfx = ' TRAIN.BATCH_SIZE_PER_GPU 8'
           else:
             epoch_sfx = ' TRAIN.END_EPOCH 3'
             batch_sfx = ''

           command = cur_python+' '+working_dir+'/tools/train.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir +' MODEL.PRETRAINED ' +pretrained_coco_model+ ' MODEL.EVALUATE False PRINT_FREQ '+str(PF)+epoch_sfx+batch_sfx
           #print(command)
           #print(xy)
           os.system(command)

        ##### inference
        annot_sfx = 'val_withJson/val/'
        sfx = 'posetrack/pose_hrnet/w48_384x288_adam_lr1e-4/final_state.pth'
        inference_model_path = out_dir + sfx
        if jj == 1:
           experiment_name = '"Baseline (# of Labeled Videos = '+V_str + '; # of Labeled Frames Per Video = '+N_str+')"'
           command = cur_python+' '+working_dir+'/tools/test.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +inference_model_path+ ' TEST.COCO_BBOX_FILE '+precomputed_boxes_file+' POSETRACK_ANNOT_DIR '+annot_dir+annot_sfx +' TEST.USE_GT_BBOX False PRINT_FREQ '+str(PF)+' EXPERIMENT_NAME ' +experiment_name
           os.system(command)
           #print(command)
           #print(xy)

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
           if rr != 3:
             epoch_sfx = ''
           else:
             epoch_sfx = ' TRAIN.END_EPOCH 19'

           command = cur_python+' '+working_dir+'/tools/train.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_train.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir +' MODEL.PRETRAINED ' +pretrained_model+ ' MODEL.EVALUATE False PRINT_FREQ '+str(PF)+epoch_sfx
           os.system(command)

        #### temporal pose aggregation
        annot_sfx = 'val_withJson/val/'
        sfx = 'posetrack/pose_hrnet/w48_384x288_adam_lr1e-4_PoseWarper_train/final_state.pth'
        pose_warper_model_path = out_dir + sfx
        if rr != 0:
         if jj == 1:
           experiment_name = '"Temporal Pose Aggregation via PoseWarper (# of Labeled Videos = '+V_str + '; # of Labeled Frames Per Video = '+N_str+')"'
           command = cur_python+' '+working_dir+'/tools/test.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_temporal_pose_aggregation.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +pose_warper_model_path+' TEST.COCO_BBOX_FILE '+precomputed_boxes_file+' POSETRACK_ANNOT_DIR '+annot_dir+annot_sfx +' TEST.USE_GT_BBOX False PRINT_FREQ '+str(PF)+' EXPERIMENT_NAME '+experiment_name
           os.system(command)
           #print(command)
           #print(xy)

        ##### video pose propagation and data augmentation experiments: inference for multiple timestep deltas
        delta_list = [-3, -2, -1, 0, 1, 2, 3]
        for dd in delta_list:
           if rr == 0:  ### video pose propagation experiments
              annot_sfx = 'val_withJson_consecutive/val/'
              if jj == 0:
                 command = cur_python+' '+working_dir+'/tools/test.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_video_pose_propagation.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +pose_warper_model_path+' POSETRACK_ANNOT_DIR '+annot_dir + annot_sfx + ' MODEL.EVALUATE False MODEL.TIMESTEP_DELTA '+str(dd) +' SAVE_PREDS True PRINT_FREQ '+str(PF)
                 os.system(command)

           elif rr != 3: ##### data augmentation experiments
              if jj == 0:
                 command = cur_python+' '+working_dir+'/tools/test.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_data_augmentation.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +pose_warper_model_path+ ' MODEL.EVALUATE False MODEL.TIMESTEP_DELTA '+str(dd)+' PRINT_FREQ '+str(PF)
                 os.system(command)
        ############

        if rr == 0: ## video pose propagation evaluation
           if jj == 1:
              #### propagating poses across video
              pseudo_gt_command = cur_python+' scripts/video_pose_propagation.py '+json_dir+' '+preds_dir
              os.system(pseudo_gt_command)

              annot_sfx = 'val_withJson_consecutive/val/'
              experiment_name = '"Video Pose Propagation from Sparsely Labeled Videos via PoseWarper"'
              command = cur_python+' '+working_dir+'/tools/test.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_video_pose_propagation.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +pose_warper_model_path+' POSETRACK_ANNOT_DIR '+annot_dir + annot_sfx + ' LOAD_PREDS True LOAD_PROPAGATED_GT_PREDS True PRINT_FREQ '+str(PF)+ ' EXPERIMENT_NAME '+experiment_name
              os.system(command)

        elif rr != 3:

           #### creating pseudo ground truth
           if jj == 0:
              pseudo_gt_command = cur_python+' scripts/augment_data_with_propagated_poses.py '+json_dir+' '+preds_dir+' '+str(V)+' '+str(N)
              os.system(pseudo_gt_command)
              #print(xy)

           ### redefine output directories
           cur_output_dir = data_aug_output_dir + 'V'+str(V)+'_N'+str(N)+'/'
           if not os.path.exists(cur_output_dir):
               os.makedirs(cur_output_dir)

           out_dir = cur_output_dir + 'out/'
           log_dir = cur_output_dir + 'log/'

           #### retrain the model with PoseWarper data augmentation
           if jj == 0:
              command = cur_python+' '+working_dir+'/tools/train.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_pseudo_gt_train.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+' DATASET.NUM_LABELED_VIDEOS '+str(V)+' DATASET.NUM_LABELED_FRAMES_PER_VIDEO '+str(N)+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir +' MODEL.PRETRAINED ' +pretrained_coco_model+ ' MODEL.EVALUATE False PRINT_FREQ '+str(PF)
              os.system(command)

           ##### inference
           annot_sfx = 'val_withJson/val/'
           sfx = 'posetrack/pose_hrnet/w48_384x288_adam_lr1e-4_pseudo_gt_train/final_state.pth'
           inference_model_path = out_dir + sfx
           if jj == 1:
              experiment_name = '"Data Augmentation via PoseWarper (# of Labeled Videos = '+str(V) + '; # of Labeled Frames Per Video = '+str(N)+')"'
              command = cur_python+' '+working_dir+'/tools/test.py --cfg experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4.yaml OUTPUT_DIR '+out_dir+' LOG_DIR '+log_dir+ ' DATASET.JSON_DIR '+json_dir +' DATASET.IMG_DIR '+img_dir+ ' TEST.MODEL_FILE ' +inference_model_path+ ' TEST.COCO_BBOX_FILE '+precomputed_boxes_file+' POSETRACK_ANNOT_DIR '+annot_dir+annot_sfx +' TEST.USE_GT_BBOX False PRINT_FREQ '+str(PF) + ' EXPERIMENT_NAME '+experiment_name
              os.system(command)
