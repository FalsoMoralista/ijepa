# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.PlantCLEF2022 import make_PlantCLEF2022

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    
    # -- make csv_logger
    csv_logger = CSVLogger(log_file,     
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)')) # TODO modify this

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    print("Target Encoder: ") 
    print(target_encoder) # TODO: Check if works.

    #mask_collator = MBMaskCollator(
    #    input_size=crop_size,
    #    patch_size=patch_size,
    #    pred_mask_scale=pred_mask_scale,
    #    enc_mask_scale=enc_mask_scale,
    #    aspect_ratio=aspect_ratio,
    #    nenc=num_enc_masks,
    #    npred=num_pred_masks,
    #    allow_overlap=allow_overlap,
    #    min_keep=min_keep)
    
    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        validation=True,
        color_jitter=color_jitter) # TODO: verify if transformations are being applied. 

    # TODO: IMPLEMENT FEATURE EXTRACTION DATASET HERE
    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_PlantCLEF2022(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler TODO: REMOVE
    #optimizer, scaler, scheduler, wd_scheduler = init_opt(
    #    encoder=encoder,
    #    predictor=predictor,
    #    wd=wd,
    #    final_wd=final_wd,
    #    start_lr=start_lr,
    #    ref_lr=lr,
    #    final_lr=final_lr,
    #    iterations_per_epoch=ipe,
    #    warmup=warmup,
    #    num_epochs=num_epochs,
    #    ipe_scale=ipe_scale,
    #    use_bfloat16=use_bfloat16)
    
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule TODO: REMOVE
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
    
    loss_meter = AverageMeter()
    time_meter = AverageMeter()

    for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(device, non_blocking=True)           
            return (imgs) 
        imgs = load_imgs()
        #maskA_meter.update(len(masks_enc[0][0])) # TODO: Remove
        #maskB_meter.update(len(masks_pred[0][0])) # TODO: Remove

        def extract_features():

            def forward_context():
                z = encoder(imgs, masks_enc) # TODO: check on masks_enc and what is it used for.
                return z

            # Step 1. Forward
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                z = forward_context() # Features extracted from the ViT context encoder
            return (z) # TODO: adjust return
        (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step) # TODO: replace by extract_features(), adjust return
        loss_meter.update(loss)
        time_meter.update(etime)

        # TODO: adjust what it is going to be logged. 
        def log_stats():
            csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
            if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                logger.info('[%d, %5d] loss: %.3f '
                            'masks: %.1f %.1f '
                            '[wd: %.2e] [lr: %.2e] '
                            '[mem: %.2e] '
                            '(%.1f ms)'
                            % (epoch + 1, itr,
                                loss_meter.avg,
                                maskA_meter.avg,
                                maskB_meter.avg,
                                _new_wd,
                                _new_lr,
                                torch.cuda.max_memory_allocated() / 1024.**2,
                                time_meter.avg))

                if grad_stats is not None:
                    logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                % (epoch + 1, itr,
                                    grad_stats.first_layer,
                                    grad_stats.last_layer,
                                    grad_stats.min,
                                    grad_stats.max))
        log_stats()
    # -- Save Checkpoint after every epoch
    logger.info('avg. loss %.3f' % loss_meter.avg)


if __name__ == "__main__":
    main()
