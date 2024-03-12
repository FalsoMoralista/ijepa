# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import shutil
import sys
from typing import Iterable, Optional
import os
import os.path as osp
import pandas as pd
import numpy as np

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, class_idx_map=None, visualize_epoch=0, max_num=30):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    save_dir = f"results/pc2022_trusted"
    '''
    # to save results
    save_dir = f"results/clef_plant_results_epoch{visualize_epoch}"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    ''' # TODO: UNCOMMENT.


    #if os.path.exists(save_dir):
    #    shutil.rmtree(save_dir)
    #    os.makedirs(save_dir)
    #else:
    #    os.makedirs(save_dir)
    # end to save result

    categories = [] # TODO: remove
    names = [] # TODO: remove
    
    #print('Class Map:', class_idx_map)
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        file_names = data_loader.dataset.samples[i * images.size(0):i * images.size(0) + images.size(0)]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

            ###################################################################
            # to save the results
            if visualize_epoch:
                scores, predictions = output.softmax(dim=1).topk(k=max_num, dim=1)
                for batch_i in range(images.size(0)):
                    file_name_ = os.path.splitext(os.path.basename(file_names[batch_i][0]))[0]
                    names.append(file_name_+'.jpg')
                    label_batch = []
                    for num in range(max_num):
                        label = predictions[batch_i, num].cpu().item()
                        #print('Predictions:', predictions)
                        #print('Label:', label)
                        label = [k for k, v in class_idx_map.items() if v == label][0]
                        #print('Label Key', label)
                        label_batch.append(label)
                        #categories.append(label) # TODO: remove
                    
                    # label = predictions[batch_i].cpu().numpy().flatten()
                    # label_batch = [k for num in range(len(label)) for k, v in class_idx_map.items() if v == label[num]]
                    score_batch = scores[batch_i].cpu().numpy()
                    rank_batch = list(range(1, max_num + 1))
                    label_batch = np.array(label_batch)

                    submission_df = pd.DataFrame({
                        'label': np.array(label_batch).flatten(),
                        'score': np.array(score_batch).flatten(),
                        'rank': np.array(rank_batch).flatten()
                    })
                    submission_df.to_csv(f"{save_dir}/{file_name_}.csv", index=False) # TODO: uncomment
            # the end
            ###################################################################

        #print('Output:', output)
        #print('Target:', target)
        if output.size(1) >= 6:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    
    #submission_df = pd.DataFrame({
    #    'Category': np.array(categories).flatten(),
    #    'Id': np.array(names).flatten()
    #})
    #submission_df.to_csv(f"{save_dir}/{'imagenet_baseline'}.csv", index=False)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_or(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate2(data_loader, model, device, class_idx_map=None, visualize_epoch=0, max_num=30):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    save_dir = f"results/imagenet_baseline"
    '''
    # to save results
    save_dir = f"results/clef_plant_results_epoch{visualize_epoch}"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    ''' # TODO: UNCOMMENT.


    #if os.path.exists(save_dir):
    #    shutil.rmtree(save_dir)
    #    os.makedirs(save_dir)
    #else:
    #    os.makedirs(save_dir)
    # end to save result

    categories = [] # TODO: remove
    names = [] # TODO: remove
    
    print('Class Map:', class_idx_map)
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        file_names = data_loader.dataset.samples[i * images.size(0):i * images.size(0) + images.size(0)]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

            ###################################################################
            # to save the results
            if visualize_epoch:
                scores, predictions = output.softmax(dim=1).topk(k=max_num, dim=1)
                for batch_i in range(images.size(0)):
                    file_name_ = os.path.splitext(os.path.basename(file_names[batch_i][0]))[0]
                    names.append(file_name_+'.jpg')
                    label_batch = []
                    for num in range(max_num):
                        label = predictions[batch_i, num].cpu().item()
                        #print('Predictions:', predictions)
                        #print('Label:', label)
                        label = [k for k, v in class_idx_map.items() if v == label][0]
                        #print('Label Key', label)
                        label_batch.append(label)
                        categories.append(label) # TODO: remove
                    
                    # label = predictions[batch_i].cpu().numpy().flatten()
                    # label_batch = [k for num in range(len(label)) for k, v in class_idx_map.items() if v == label[num]]
                    score_batch = scores[batch_i].cpu().numpy()
                    rank_batch = list(range(1, max_num + 1))
                    label_batch = np.array(label_batch)

                    submission_df = pd.DataFrame({
                        'label': np.array(label_batch).flatten(),
                        'score': np.array(score_batch).flatten(),
                        'rank': np.array(rank_batch).flatten()
                    })
                    #submission_df.to_csv(f"{save_dir}/{file_name_}.csv", index=False) # TODO: uncomment
            # the end
            ###################################################################

        #print('Output:', output)
        #print('Target:', target)
        if output.size(1) >= 6:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    
    submission_df = pd.DataFrame({
        'Category': np.array(categories).flatten(),
        'Id': np.array(names).flatten()
    })
    submission_df.to_csv(f"{save_dir}/{'imagenet_baseline'}.csv", index=False)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}