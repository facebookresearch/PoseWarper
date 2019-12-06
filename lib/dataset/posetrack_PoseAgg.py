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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path as osp

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from dataset.JointsDataset_PoseAgg import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms

import scipy.optimize
import scipy.spatial
import scipy.io as sio

import pickle as pkl


logger = logging.getLogger(__name__)

coco_src_keypoints = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle']
posetrack_src_keypoints = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle']
dst_keypoints = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'nose',
    'head_top']

def compute_head_size(kps):
    head_top = kps[:2, 2]
    head_bottom = kps[:2, 1]
    # originally was 0.6 x hypotenuse of the head, but don't have those kpts
    return np.linalg.norm(head_top - head_bottom) + 1  # to avoid 0s


def pck_distance(kps_a, kps_b, dist_thresh=0.5):
    """
    Compute distance between the 2 keypoints, where each is represented
    as a 3x17 or 4x17 np.ndarray.
    Ideally use kps_a as GT, if one is GT
    """
    head_size = compute_head_size(kps_a)
    # distance between all points
    # TODO(rgirdhar): Some of these keypoints might be invalid -- not labeled
    # in the dataset. Might need to do something about that...
    normed_dist = np.linalg.norm(kps_a[:2] - kps_b[:2], axis=0) / head_size
    match = normed_dist < dist_thresh
    pck = np.sum(match) / match.size
    pck_dist = 1.0 - pck
    return pck_dist

def _compute_pairwise_bb_distance(a, b):
    res = np.ones((len(a), len(b)))
    th = 0.3
    for i in range(len(a)):
        kp1 = a[i]
        idx1 = np.where(kp1[2, :] > th)[0]
        if len(idx1)>0:
          temp1 = kp1[:, idx1]
          x11 = min(temp1[0,:])
          x12 = max(temp1[0,:])
          y11 = min(temp1[1,:])
          y12 = max(temp1[1,:])
          boxA = [x11, y11, x12, y12]

          for j in range(len(b)):

              kp2 = b[j]
              idx2 = np.where(kp2[2, :] > th)[0]
              if len(idx2)>0:
                temp2 = kp2[:, idx2]
                x21 = min(temp2[0,:])
                x22 = max(temp2[0,:])
                y21 = min(temp2[1,:])
                y22 = max(temp2[1,:])
                boxB = [x21, y21, x22, y22]

                # determine the (x, y)-coordinates of the intersection rectangle
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])

                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)

                res[i, j] = 1 - iou

    # return the intersection over union value
    return res

def _compute_pairwise_kpt_distance(a, b):
    """
    Args:
        a, b (poses): Two sets of poses to match
        Each "poses" is represented as a list of 3x17 or 4x17 np.ndarray
    """
    res = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            res[i, j] = pck_distance(a[i], b[j])
    return res



def video2filenames(annot_dir):
    pathtodir = annot_dir


    output = {}
    L = {}
    mat_files = [f for f in os.listdir(pathtodir) if
             osp.isfile(osp.join(pathtodir, f)) and '.mat' in f]
    json_files = [f for f in os.listdir(pathtodir) if
             osp.isfile(osp.join(pathtodir, f)) and '.json' in f]

    if len(json_files) > 1:
        files = json_files
        ext_types = '.json'
    else:
        files = mat_files
        ext_types = '.mat'

    for fname in files:
        if ext_types == '.mat':
            out_fname = fname.replace('.mat', '.json')
            data = sio.loadmat(
                osp.join(pathtodir, fname), squeeze_me=True,
                struct_as_record=False)
            temp = data['annolist'][0].image.name


            data2 = sio.loadmat(osp.join(pathtodir, fname))
            num_frames = len(data2['annolist'][0])
        elif ext_types == '.json':
            out_fname = fname
            with open(osp.join(pathtodir, fname), 'r') as fin:
                data = json.load(fin)

            if 'annolist' in data:
              temp = data['annolist'][0]['image'][0]['name']
              num_frames = len(data['annolist'])
            else:
              temp = data['images'][0]['file_name']
              num_frames = data['images'][0]['nframes']


        else:
            raise NotImplementedError()
        video = osp.dirname(temp)
        output[video] = out_fname
        L[video] = num_frames
    return output, L

def coco2posetrack(preds, src_kps, dst_kps, global_score):
    data = []
    global_score = float(global_score)
    dstK = len(dst_kps)
    for k in range(dstK):

        if dst_kps[k] in src_kps:
            ind = src_kps.index(dst_kps[k])
            local_score = (preds[2, ind] + preds[2, ind]) / 2.0
            #conf = global_score
            conf = local_score*global_score
            #if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({'id': [k],
                             'x': [float(preds[0, ind])],
                             'y': [float(preds[1, ind])],
                             'score': [conf]})
        elif dst_kps[k] == 'neck':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            #conf_msho = global_score
            conf_msho = local_score*global_score

            #if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({'id': [k],
                             'x': [float(x_msho)],
                             'y': [float(y_msho)],
                             'score': [conf_msho]})
        elif dst_kps[k] == 'head_top':
            #print(xy)
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')


            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0


            nose = src_kps.index('nose')
            x_nose = preds[0, nose]
            y_nose = preds[1, nose]
            x_tophead = x_nose - (x_msho - x_nose)
            y_tophead = y_nose - (y_msho - y_nose)
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0

            #if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({
                    'id': [k],
                    'x': [float(x_tophead)],
                    'y': [float(y_tophead)],
                    'score': [conf_htop]})
    return data


class PoseTrackDataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):


        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.img_dir = cfg.DATASET.IMG_DIR
        self.json_dir = cfg.DATASET.JSON_DIR

        self.num_labeled_frames_per_video = cfg.DATASET.NUM_LABELED_FRAMES_PER_VIDEO
        self.num_labeled_videos = cfg.DATASET.NUM_LABELED_VIDEOS
        self.pseudo_gt_train = cfg.DATASET.PSEUDO_GT_TRAIN
        self.num_adjacent_frames = int(cfg.DATASET.NUM_ADJACENT_FRAMES)

        self.timestep_delta = int(cfg.MODEL.TIMESTEP_DELTA)

        self.use_gt_input_test = cfg.MODEL.USE_GT_INPUT_TEST
        self.warping_reverse = cfg.MODEL.WARPING_REVERSE
        self.load_preds = cfg.LOAD_PREDS
        self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        self.json_file = cfg.DATASET.JSON_FILE
        self.flow_type = cfg.FLOW_TYPE
        self.only_consecutive = cfg.ONLY_CONSECUTIVE
        self.only_consecutive_val_finetune = cfg.ONLY_CONSECUTIVE_VAL_FINETUNE

        self.eval_tracking = cfg.EVAL_TRACKING
        self.tracking_threshold = cfg.TRACKING_THRESHOLD


        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17
        self.flip_pairs = [[3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.is_train = is_train

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """

        if self.is_train:
           if self.num_labeled_frames_per_video <0:
             return self.json_dir + 'posetrack_train.json'
           else:
             if self.pseudo_gt_train:
                if self.num_labeled_videos != 250:
                    return self.json_dir + 'posetrack_train_N'+str(self.num_labeled_videos)+'_per_video_V'+str(self.num_labeled_frames_per_video)+'_videos_wPseudoGT.json'
                else:
                    return self.json_dir + 'posetrack_train_N'+str(self.num_labeled_frames_per_video)+'_per_video_wPseudoGT.json'

             else:
                 if self.num_labeled_videos == 250:
                    return self.json_dir + 'posetrack_train_N'+str(self.num_labeled_frames_per_video)+'_per_video.json'
                 else:
                    return self.json_dir + 'posetrack_train_N'+str(self.num_labeled_frames_per_video)+'_per_video_V'+str(self.num_labeled_videos)+'_videos.json'
        else:
             if self.json_file!='':
                return self.json_file

             if self.num_labeled_frames_per_video > 0 and self.test_on_train:
                 if self.num_labeled_videos == 250:
                    return self.json_dir + 'posetrack_train_N'+str(self.num_labeled_frames_per_video)+'_per_video.json'
                 else:
                    return self.json_dir + 'posetrack_train_N'+str(self.num_labeled_frames_per_video)+'_per_video_V'+str(self.num_labeled_videos)+'_videos.json'

             elif self.only_consecutive:
                 return self.json_dir + 'posetrack_val_consecutive.json'
             else:
                 return self.json_dir + 'posetrack_val.json'

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_posetrack_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        file_name = im_ann['file_name']

        nframes = int(im_ann['nframes'])
        frame_id = int(im_ann['frame_id'])

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:

            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.img_dir + file_name,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _load_posetrack_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': self.img_dir + img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'nframes': nframes,
                'frame_id': frame_id,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db



    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):

        if 'train' in cfg.POSETRACK_ANNOT_DIR:
           if cfg.LOAD_PREDS:
             output_dir = cfg.OUTPUT_DIR + 'json_results_train_delta'+str(cfg.MODEL.TIMESTEP_DELTA)+'/'
           else:
             output_dir = cfg.OUTPUT_DIR + 'json_results_train/'
        else:
           output_dir = cfg.OUTPUT_DIR + 'json_results_val/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ### processing our preds
        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        all_tracks = []
        cc = 0

        #print(img_path)
        for key in img_path:
            temp = key.split('/')

            #video_name = osp.dirname(key)
            video_name = temp[len(temp)-3] + '/' + temp[len(temp)-2]
            img_sfx = temp[len(temp)-3] + '/' + temp[len(temp)-2] + '/' + temp[len(temp)-1]


            prev_nm = temp[len(temp)-1]
            frame_num = int(prev_nm.replace('.jpg',''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            for idx in idx_list:
                temp = np.zeros((4,17))
                temp[0,:] = preds[idx,:,0]
                temp[1,:] = preds[idx,:,1]
                temp[2,:] = preds[idx,:,2]
                temp[3,:] = preds[idx,:,2]
                pose_list.append(temp)


                temp = np.zeros((1,6))
                temp[0,:] = boxes[idx,:]
                box_list.append(temp)


            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc +=1

        annot_dir  = cfg.POSETRACK_ANNOT_DIR
        is_posetrack18 = cfg.DATASET.IS_POSETRACK18

        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            cur_length = L['images/'+vid]

            temp_kps_map = {}
            temp_track_kps_map = {}
            temp_box_map = {}

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]
                img_sfx = vid2name_map[vid][c]
                c+=1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb
            #### including empty frames
            nnz_counter = 0
            next_track_id = 0

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length

            for jj in range(sid,fid):
                frame_num = jj
                if not jj in used_frame_list:
                    temp_sfx = vid2name_map[vid][0]
                    arr = temp_sfx.split('/')
                    if not is_posetrack18:
                       img_sfx = arr[0]+'/'+arr[1]+'/'+str(frame_num).zfill(8)+'.jpg'
                    else:
                       img_sfx = arr[0]+'/'+arr[1]+'/'+str(frame_num).zfill(6)+'.jpg'
                    kps = []
                    tracks = []
                    bbs = []

                else:

                    img_sfx = temp_kps_map[frame_num][0]
                    kps = temp_kps_map[frame_num][1]
                    bbs = temp_box_map[frame_num]
                    tracks = [1] * len(kps)

                ### creating a data element
                data_el = {
                    'image': {'name':img_sfx},
                    'imgnum': [frame_num],
                    'annorect': self._convert_data_to_annorect_struct(kps, tracks, bbs),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        #### saving files for evaluation
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(
                output_dir, out_filenames[osp.join('images', vname)])
            with open(outfpath, 'w') as fout:
                json.dump({'annolist': vdata}, fout)

        #run evaluation
        AP = self._run_eval(annot_dir, output_dir)
        name_value = [
            ('Head', AP[0]),
            ('Shoulder', AP[1]),
            ('Elbow', AP[2]),
            ('Wrist', AP[3]),
            ('Hip', AP[4]),
            ('Knee', AP[5]),
            ('Ankle', AP[6]),
            ('Mean', AP[7])
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']


    def _convert_data_to_annorect_struct(self, poses, tracks, boxes):
        """
        Args:
            boxes (np.ndarray): Nx5 size matrix with boxes on this frame
            poses (list of np.ndarray): N length list with each element as 4x17 array
            tracks (list): N length list with track ID for each box/pose
        """
        num_dets = len(poses)
        annorect = []
        for j in range(num_dets):
            score = boxes[j][0,5]
            if self.eval_tracking and score < self.tracking_threshold:
                continue
            point = coco2posetrack(
                poses[j], posetrack_src_keypoints, dst_keypoints, score)
            annorect.append({'annopoints': [{'point': point}],
                             'score': [float(score)],
                             'track_id': [tracks[j]]})
        if num_dets == 0:
            # MOTA requires each image to have at least one detection! So, adding
            # a dummy prediction.
            annorect.append({
                'annopoints': [{'point': [{
                    'id': [0],
                    'x': [0],
                    'y': [0],
                    'score': [-100.0],
                }]}],
                'score': [0],
                'track_id': [0]})
        return annorect



    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _run_eval(self,annot_dir, output_dir, eval_tracking=False, eval_pose=True):
        """
        Runs the evaluation, and returns the "total mAP" and "total MOTA"
        """
        from poseval.py import evaluate_simple
        eval_tracking = self.eval_tracking
        apAll = evaluate_simple.evaluate(
            annot_dir, output_dir, eval_pose, eval_tracking)

        return apAll
