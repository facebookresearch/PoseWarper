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
import cv2
import pickle as pkl
import scipy.io
import h5py

### inputs
json_dir = sys.argv[1]
preds_dir = sys.argv[2]
dd_list = [-3, -2, -1, 1, 2, 3]

######
gt_preds_file = preds_dir + 'keypoint_preds/delta0_keypoints_reverse.h5'
gt_boxes_file = preds_dir + 'keypoint_preds/delta0_boxes_reverse.h5'

output_file = preds_dir + 'keypoint_preds/propagated_gt_preds.h5'
output_bb_file = preds_dir + 'keypoint_preds/propagated_gt_boxes.h5'
output_filenames_map_file = preds_dir + 'keypoint_preds/filenames_map.npy'

gt_file = json_dir + 'posetrack_val_consecutive.json'
with open(gt_file) as f:
    gt_data = json.load(f)


#temp = scipy.io.loadmat(gt_preds_file)
#kps0_gt_preds = temp['data']
#
#temp = scipy.io.loadmat(gt_boxes_file)
#bb0_gt_preds = temp['data']

hf = h5py.File(gt_preds_file, 'r')
kps0_gt_preds = np.array(hf.get('data'))
hf.close()

hf = h5py.File(gt_boxes_file, 'r')
bb0_gt_preds = np.array(hf.get('data'))
hf.close()

im_list= gt_data['images']
gt_list = gt_data['annotations']


### finding valid indices
c = 0
valid_idx = []
im_name_list = []
frame_id_list = []
video_length_list = []

nm2id_map ={}
id2nm_map ={}

video2seq_map = {}
best_video2seq_map = {}


seq2id_map = {}

mm = {}
cc = 0
for i in range(len(gt_list)):
    im_idx = gt_list[i]['image_id'] - 1
    im_el = im_list[im_idx]
    height = im_el['height']
    width = im_el['width']

    cur_image_name = im_el['file_name']
    temp = cur_image_name.split('/')
    video_nm = temp[0] + '/' +temp[1]

    nframes = im_el['nframes']
    frame_id = im_el['frame_id']

    if not cur_image_name in mm:
      if not video_nm in video2seq_map:
          video2seq_map[video_nm] = [frame_id]
      else:
          prev_frame_id = video2seq_map[video_nm][-1]
          if prev_frame_id + 1 != frame_id:
              if not video_nm in best_video2seq_map:
                  best_video2seq_map[video_nm] = video2seq_map[video_nm]
              else:
                  if len(video2seq_map[video_nm]) > len(best_video2seq_map[video_nm]):
                     best_video2seq_map[video_nm] = video2seq_map[video_nm]
              video2seq_map[video_nm] = [frame_id]

          else:
              video2seq_map[video_nm].append(frame_id)
    ###
    mm[cur_image_name] =1


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
        im_name_list.append(cur_image_name)
        frame_id_list.append(frame_id)
        video_length_list.append(nframes)

        cur_key = (video_nm, frame_id)
        if not cur_key in seq2id_map:
            seq2id_map[cur_key] = [cc]
        else:
            seq2id_map[cur_key].append(cc)

        if not cur_image_name in nm2id_map:
            nm2id_map[cur_image_name] = [cc]
        else:
            nm2id_map[cur_image_name].append(cc)

        id2nm_map[cc] = cur_image_name
        cc +=1


for video_nm in video2seq_map:
  if not video_nm in best_video2seq_map:
      best_video2seq_map[video_nm] = video2seq_map[video_nm]
  else:
      if len(video2seq_map[video_nm]) > len(best_video2seq_map[video_nm]):
         best_video2seq_map[video_nm] = video2seq_map[video_nm]

#################3



N = len(valid_idx)

#print(N)
#print(kps0_gt_preds.shape)
assert(N == kps0_gt_preds.shape[0])

############
preds_map = {}
boxes_map = {}
for dd in dd_list:

     cur_preds_file = preds_dir + 'keypoint_preds/delta'+str(dd)+'_keypoints_reverse.h5'
     hf = h5py.File(cur_preds_file, 'r')
     kps_sup_preds = np.array(hf.get('data'))
     hf.close()
     preds_map[dd] = kps_sup_preds

     #########3
     cur_boxes_file = preds_dir + 'keypoint_preds/delta'+str(dd)+'_boxes_reverse.h5'
     hf = h5py.File(cur_boxes_file, 'r')
     bb_sup_preds = np.array(hf.get('data'))
     hf.close()
     boxes_map[dd] = bb_sup_preds
##########333



########33
pred_idx_map = {}
for i in range(N):
  cur_frame_id = frame_id_list[i]

  cur_im_name = im_name_list[i]
  temp = cur_im_name.split('/')
  video_nm = temp[0] + '/' + temp[1]

  for dd in dd_list:
     sup_frame_id = cur_frame_id + dd

     key = (video_nm, cur_frame_id, sup_frame_id)
     if not key in pred_idx_map:
         pred_idx_map[key] = [i]
     else:
         pred_idx_map[key].append(i)

covered_idx = {}
covered_frame2idx = {}
covered_frame2delta = {}
visited_seqs = {}
visited_ids = {}

for video_nm in best_video2seq_map:
   covered_frame2idx[video_nm] = {}
   covered_frame2delta[video_nm] = {}

   seq = best_video2seq_map[video_nm]

   if len(seq) <= 3:
      for s in seq:
          missing += seq2id_map[(video_nm,s)]

   if len(seq) > 3:
     sid = seq[3]
     fid = seq[-1]
     gg = 7
     labeled_frame_idx = [sid]
     while sid <= fid:
         sid += gg
         if sid <= fid:
            labeled_frame_idx.append(sid)

     if fid - labeled_frame_idx[-1] > 3:
         labeled_frame_idx.append(fid)

     #####
     visited = {}
     for l in labeled_frame_idx:
        #### delta = 0
        cur_im_file = video_nm +'/' + str(l).zfill(8)+'.jpg'
        #print(nm2id_map.keys())
        delta0_idx_list = nm2id_map[cur_im_file]

        covered_frame2delta[video_nm][l] = 0
        covered_frame2idx[video_nm][l] = delta0_idx_list

        ####### non-zero deltas
        for dd in dd_list:
           cur_frame_id = l
           sup_frame_id = cur_frame_id + dd

           if sup_frame_id in seq:

              key = (video_nm, cur_frame_id, sup_frame_id)
              #if key in pred_idx_map:
              if not sup_frame_id in covered_frame2delta[video_nm] or abs(dd) < abs(covered_frame2delta[video_nm][sup_frame_id]):

                 cur_idx_list = pred_idx_map[key]

                 covered_frame2delta[video_nm][sup_frame_id] = dd
                 covered_frame2idx[video_nm][sup_frame_id] = cur_idx_list


     ### making sure that everything was covered
     for s in seq:
         if not s in covered_frame2delta[video_nm]:
             raise Exception('Frame '+str(s)+ ' in Video ' +video_nm + ' was not covered!!')
         visited_seqs[(video_nm,s)] = 1


propagated_gt_preds = np.zeros((N, 17, 3))
propagated_gt_boxes = np.zeros((N,6))

filenames_map = {}

#fp = open('nm2delta_map.txt','w')

cc = 0
for video_nm in covered_frame2idx:
    S = covered_frame2idx[video_nm]
    for s in S:
       idx_list = covered_frame2idx[video_nm][s]
       dd = covered_frame2delta[video_nm][s]

       cur_im_file = video_nm +'/' + str(s).zfill(8)+'.jpg'

       for j in idx_list:
          if not cur_im_file in filenames_map:
              filenames_map[cur_im_file] = [cc]
          else:
              filenames_map[cur_im_file].append(cc)
          ##------
          if dd == 0:
             cur_pred = kps0_gt_preds[j,:,:]
             cur_bb = bb0_gt_preds[j,:]
          else:
             cur_pred = preds_map[dd][j,:,:]
             cur_bb = boxes_map[dd][j,:]

#          fp.write(str(cc)+','+str(dd)+','+str(j)+'\n')
#          fp.write(str(cur_im_file)+','+str(dd)+'\n')

#
#          if cc == 0:
#              print(cur_pred)
#              print(xy)

          ### filling the predictions
          propagated_gt_preds[cc,:,:] = cur_pred
          propagated_gt_boxes[cc,:] = cur_bb
          cc += 1

#fp.close()
#print(cc)
#print(N)
#print(xy)

#### saving preds
#scipy.io.savemat(output_file,{'data': propagated_gt_preds})
#scipy.io.savemat(output_bb_file,{'data': propagated_gt_boxes})

#print(propagated_gt_preds[0,2,:])
#print(xy)

hf = h5py.File(output_file, 'w')
hf.create_dataset('data', data=propagated_gt_preds)
hf.close()

hf = h5py.File(output_bb_file, 'w')
hf.create_dataset('data', data=propagated_gt_boxes)
hf.close()

np.save(output_filenames_map_file, filenames_map)
