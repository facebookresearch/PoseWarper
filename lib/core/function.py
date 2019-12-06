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

import time
import logging
import os

import numpy as np
import torch
import scipy.io
import torch.nn as nn
import h5py

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import json

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    use_warping = config['MODEL']['USE_WARPING_TRAIN']
    use_gt_input = config['MODEL']['USE_GT_INPUT_TRAIN']

    N = min(len(train_loader),config['MODEL']['ITER'])

    if use_warping:
      for i, (input, input_sup,  target, target_weight, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        ### concatenating
        if use_gt_input:
           target_up_op  = nn.Upsample(scale_factor=4, mode='nearest')
           target_up = target_up_op(target)
           concat_input = torch.cat((input, input_sup, target_up), 1)
        else:
           concat_input = torch.cat((input, input_sup), 1)

        # compute output
        outputs = model(concat_input)
        #outputs = model(input)


        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
    else:
      for i, (input,  target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)


        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    filenames_map = {}
    filenames_counter = 0
    imgnums = []
    idx = 0

    ############3
    preds_output_dir = config.OUTPUT_DIR + 'keypoint_preds/'
    if config.SAVE_PREDS:
        output_filenames_map_file = preds_output_dir + 'filenames_map.npy'
        if not os.path.exists(preds_output_dir):
            os.makedirs(preds_output_dir)
    ####################


    use_warping = config['MODEL']['USE_WARPING_TEST']
    use_gt_input = config['MODEL']['USE_GT_INPUT_TEST']
    warping_reverse = config['MODEL']['WARPING_REVERSE']

    ####################################################
    if config.LOAD_PROPAGATED_GT_PREDS:
       output_path = preds_output_dir + 'propagated_gt_preds.h5'
       hf = h5py.File(output_path, 'r')
       all_preds = np.array(hf.get('data'))
       hf.close()

       output_path = preds_output_dir + 'propagated_gt_boxes.h5'
       hf = h5py.File(output_path, 'r')
       all_boxes = np.array(hf.get('data'))
       hf.close()

       output_path = preds_output_dir + 'filenames_map.npy'
       D=np.load(output_path, allow_pickle=True)
       filenames_map = D.item()

       track_preds = None
       logger.info('########################################')
       logger.info('{}'.format(config.EXPERIMENT_NAME))
       name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, filenames_map, track_preds, filenames, imgnums)

       model_name = config.MODEL.NAME
       if isinstance(name_values, list):
           for name_value in name_values:
               _print_name_value(name_value, model_name)
       else:
           _print_name_value(name_values, model_name)

       if writer_dict:
           writer = writer_dict['writer']
           global_steps = writer_dict['valid_global_steps']
           writer.add_scalar(
               'valid_loss',
               losses.avg,
               global_steps
           )
           writer.add_scalar(
               'valid_acc',
               acc.avg,
               global_steps
           )
           if isinstance(name_values, list):
               for name_value in name_values:
                   writer.add_scalars(
                       'valid',
                       dict(name_value),
                       global_steps
                   )
           else:
               writer.add_scalars(
                   'valid',
                   dict(name_values),
                   global_steps
               )
           writer_dict['valid_global_steps'] = global_steps + 1

       return perf_indicator
    ###################################3


    with torch.no_grad():
      end = time.time()
      if not use_warping:
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            ########
            for ff in range(len(meta['image'])):
                cur_nm = meta['image'][ff]

                if not cur_nm in filenames_map:
                    filenames_map[cur_nm] = [filenames_counter]
                else:
                    filenames_map[cur_nm].append(filenames_counter)
                filenames_counter +=1
            #########


            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)


            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images


            if i % config.PRINT_FREQ == 0:

                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        track_preds = None
        logger.info('########################################')
        logger.info('{}'.format(config.EXPERIMENT_NAME))
        name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, filenames_map, track_preds, filenames, imgnums)

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

      else: ### PoseWarper
        for i, (input, input_sup, target, target_weight, meta) in enumerate(val_loader):

            for ff in range(len(meta['image'])):
                cur_nm = meta['image'][ff]
                if not cur_nm in filenames_map:
                    filenames_map[cur_nm] = [filenames_counter]
                else:
                    filenames_map[cur_nm].append(filenames_counter)
                filenames_counter +=1

            ### concatenating
            if use_gt_input:
               target_up_op  = nn.Upsample(scale_factor=4, mode='nearest')
               target_up = target_up_op(target)
               concat_input = torch.cat((input, input_sup, target_up), 1)
            else:
               if warping_reverse:
                  target_up_op  = nn.Upsample(scale_factor=4, mode='nearest')
                  target_up = target_up_op(target)
                  concat_input = torch.cat((input, input_sup, target_up), 1)
               else:
                  concat_input = torch.cat((input, input_sup), 1)
            ###########

            if not config.LOAD_PREDS:
                 outputs = model(concat_input)

                 if isinstance(outputs, list):
                     output = outputs[-1]
                 else:
                     output = outputs

                 target = target.cuda(non_blocking=True)
                 target_weight = target_weight.cuda(non_blocking=True)


            num_images = input.size(0)

            if config.LOAD_PREDS:
                loss = 0.0
                avg_acc = 0.0
                cnt = 1
            else:
                loss = criterion(output, target, target_weight)
                losses.update(loss.item(), num_images)

                # measure accuracy and record loss
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            if not config.LOAD_PREDS:
                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score


            ##############
            image_path.extend(meta['image'])


            idx += num_images

            if i % config.PRINT_FREQ == 0:

                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )

                if not config.LOAD_HEATMAPS and not config.LOAD_PREDS:
                   save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        if config.SAVE_PREDS:
           print('Saving preds...')
           output_path = preds_output_dir + 'delta'+str(config.MODEL.TIMESTEP_DELTA)+'_keypoints.h5'
#           output_path = preds_output_dir + 'delta'+str(config.MODEL.TIMESTEP_DELTA)+'_th'+str(config.TEST.IMAGE_THRE)+'_keypoints.h5'
           if config.MODEL.WARPING_REVERSE:
               output_path = output_path.replace('.h5','_reverse.h5')

           if config.DATASET.TEST_ON_TRAIN:
               output_path = output_path.replace('.h5','_train.h5')

           print(output_path)
           hf = h5py.File(output_path, 'w')
           hf.create_dataset('data', data=all_preds)
           hf.close()

           output_path = preds_output_dir + 'delta'+str(config.MODEL.TIMESTEP_DELTA)+'_boxes.h5'
#           output_path = preds_output_dir + 'delta'+str(config.MODEL.TIMESTEP_DELTA)+'_th'+str(config.TEST.IMAGE_THRE)+'_boxes.h5'
           if config.MODEL.WARPING_REVERSE:
               output_path = output_path.replace('.h5','_reverse.h5')

           if config.DATASET.TEST_ON_TRAIN:
               output_path = output_path.replace('.h5','_train.h5')
           hf = h5py.File(output_path, 'w')
           hf.create_dataset('data', data=all_boxes)
           hf.close()

#           if config.MODEL.TIMESTEP_DELTA == 0:
#             output_filenames_map_file = output_filenames_map_file.replace('.npy','_th'+str(config.TEST.IMAGE_THRE)+'.npy')
#             print(output_filenames_map_file)
#             np.save(output_filenames_map_file, filenames_map)


        if config.LOAD_PREDS:
           #print('Loading preds...')
           output_path = preds_output_dir + 'delta'+str(config.MODEL.TIMESTEP_DELTA)+'_keypoints'+sfx+'.h5'
           hf = h5py.File(output_path, 'r')
           all_preds = np.array(hf.get('data'))
           hf.close()

           output_path = preds_output_dir + 'delta'+str(config.MODEL.TIMESTEP_DELTA)+'_boxes'+sfx+'.h5'
           hf = h5py.File(output_path, 'r')
           all_boxes = np.array(hf.get('data'))
           hf.close()
        ####################

        if config.MODEL.EVALUATE:
           track_preds = None
           logger.info('########################################')
           logger.info('{}'.format(config.EXPERIMENT_NAME))
           name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, filenames_map, track_preds, filenames, imgnums)

           model_name = config.MODEL.NAME
           if isinstance(name_values, list):
               for name_value in name_values:
                   _print_name_value(name_value, model_name)
           else:
               _print_name_value(name_values, model_name)

           if writer_dict:
               writer = writer_dict['writer']
               global_steps = writer_dict['valid_global_steps']
               writer.add_scalar(
                   'valid_loss',
                   losses.avg,
                   global_steps
               )
               writer.add_scalar(
                   'valid_acc',
                   acc.avg,
                   global_steps
               )
               if isinstance(name_values, list):
                   for name_value in name_values:
                       writer.add_scalars(
                           'valid',
                           dict(name_value),
                           global_steps
                       )
               else:
                   writer.add_scalars(
                       'valid',
                       dict(name_values),
                       global_steps
                   )
               writer_dict['valid_global_steps'] = global_steps + 1
        else:
            perf_indicator = None

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
