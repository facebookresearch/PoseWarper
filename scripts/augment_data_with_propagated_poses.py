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
import json
from pprint import pprint
import numpy as np
import copy
import scipy.io
import h5py

#### inputs
json_dir = sys.argv[1]
preds_dir = sys.argv[2]
V = int(sys.argv[3])
N = int(sys.argv[4])

T = 3
dd_list = [-3, -2, -1, 1, 2, 3]
################3

if V==250:
  input_file = json_dir + 'posetrack_train_N1_per_video.json'
  preds_file = preds_dir + 'keypoint_preds/delta1_keypoints_reverse_train.h5'
  output_file = json_dir + 'posetrack_train_N1_per_video_wPseudoGT.json'
else:
  input_file = json_dir + 'posetrack_train_N'+str(N)+'_per_video_V'+str(V)+'_videos.json'
  preds_file = preds_dir + 'keypoint_preds/delta1_keypoints_reverse_train.h5'
  output_file = json_dir + 'posetrack_train_N'+str(N)+'_per_video_V'+str(V)+'_videos_wPseudoGT.json'
#############3

with open(input_file) as f:
    data = json.load(f)

im_list = data['images']
gt_list = data['annotations']

print('# of images BEFORE augmentation: '+str(len(data['images'])))
print('# of pose instances BEFORE augmentation: '+str(len(data['annotations'])))

next_im_id = im_list[len(im_list)-1]['id'] + 1
next_gt_id = gt_list[len(gt_list)-1]['id'] + 1

### finding valid indices
c = 0
valid_idx = []
for i in range(len(gt_list)):
    im_idx = gt_list[i]['image_id'] - 1
    im_el = im_list[im_idx]
    height = im_el['height']
    width = im_el['width']

    kps = gt_list[i]['keypoints']
    A = gt_list[i]['area']
    x, y, w, h = gt_list[i]['bbox']
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    v = max(kps)
    if v != 0 and A > 0 and x2 >= x1 and y2 >= y1:
        valid_idx.append(i)
###################3

missing = 0

new_im_map = {}
for dd in dd_list:
    cur_preds_file = preds_file.replace('delta1','delta'+str(dd))
    #temp = scipy.io.loadmat(cur_preds_file)
    #preds = temp['data']
    hf = h5py.File(cur_preds_file, 'r')
    preds = np.array(hf.get('data'))
    hf.close()


    assert(len(valid_idx) == preds.shape[0])

    c = 0
    for i in valid_idx:
        im_idx = gt_list[i]['image_id'] - 1

        gt_el = copy.deepcopy(gt_list[i])
        im_el = copy.deepcopy(im_list[im_idx])

        kps_list = gt_list[i]['keypoints']
        cur_joints = np.zeros((17,3))
        for k in range(17):
            x = kps_list[k*3]
            y = kps_list[k*3+1]
            score = kps_list[k*3+2]

            cur_joints[k, 0] = x
            cur_joints[k, 1] = y
            cur_joints[k, 2] = score
        ############

        new_joints = np.squeeze(preds[c,:,:])
        c +=1

        new_kps_list = []
        for j in range(17):
           if new_joints[j,0] > 0 and new_joints[j,1]>0 and new_joints[j,2]> 0.2:
             new_kps_list.append(float(new_joints[j,0]))
             new_kps_list.append(float(new_joints[j,1]))
             new_kps_list.append(2.0)

           else:
             new_kps_list.append(0.0)
             new_kps_list.append(0.0)
             new_kps_list.append(0.0)

        ### changing Image element
        temp = im_el['file_name'].split('/')


        old_frame_nm = temp[len(temp)-1]
        new_frame_id = im_el['frame_id'] + dd
        new_frame_nm = str(new_frame_id).zfill(8) + '.jpg'
        im_el['file_name'] = im_el['file_name'].replace(old_frame_nm,new_frame_nm)
        im_el['original_file_name'] = im_el['original_file_name'].replace(old_frame_nm,new_frame_nm)
        im_el['id'] = next_im_id
        im_el['frame_id'] = new_frame_id

        ### appending new image element structure
        cur_key = im_el['original_file_name']
        if not cur_key in new_im_map:
            new_im_map[cur_key] = next_im_id
            data['images'].append(im_el)
            next_im_id +=1


        #### changing GT element
        gt_el['keypoints'] = new_kps_list
        gt_el['head_box'] = []
        gt_el['id'] = next_gt_id
        gt_el['image_id'] = new_im_map[cur_key]

        #### apending GT list
        data['annotations'].append(gt_el)
        next_gt_id +=1

print('------------')
print('# of images AFTER augmentation: '+str(len(data['images'])))
print('# of pose instances AFTER augmentation: '+str((len(data['annotations']))))
print('------------')
print('Saving to:')
print(output_file)
with open(output_file, 'w') as f:
    json.dump(data, f)
