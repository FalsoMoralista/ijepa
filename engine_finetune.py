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
    add_classification_head,
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# -- MAE
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


# --
log_timings = True
log_freq = 5
checkpoint_freq = 5
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


    drop_path = args['data']['drop_path']
    mixup = args['data']['mixup']
    cutmix = args['data']['cutmix']
    reprob = args['data']['reprob']
    nb_classes = args['data']['nb_classes']

    eval_epoch = args['optimization']['eval_epoch'] # TODO CHECK where it is used []

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
    smoothing = args['optimization']['label_smoothing']
    

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
        load_path = '/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/' + 'IN1K-vit.h.14-300e.pth.tar' #os.path.join(folder, r_file) if r_file is not None else latest_path
    
    #if load_model:
    #    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'train loss'),
                           ('%.5f', 'test loss'),
                           ('%.3f', 'Train - Acc@1'),
                           ('%.3f', 'Train - Acc@5'),
                           ('%.3f', 'Test - Acc@1'),
                           ('%.3f', 'Test - Acc@5'),
                           ('%d', 'time (ms)'))
    
    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    # TODO: 
    # Adjust []
    # Add RandAugment(9, 0.5) []
    transform = make_transforms( 
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        normalization=((0.436, 0.444, 0.330),
                (0.203, 0.199, 0.195)), # PlantCLEF normalization
        color_jitter=color_jitter)

    # TODO: Create supervised loaders []
    # (1) - Build train and val datasets as below.
    # (2) - Build Train and Val data loaders upon the datasets.
    # (3) - 
    #dataset_train = build_dataset(is_train=True, args=args)
    #dataset_val = build_dataset(is_train=False, args=args)


    # -- init data-loaders/samplers
    _, supervised_loader_train, supervised_sampler_train = make_PlantCLEF2022(
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
            supervision=True,
            drop_last=True)
    ipe = len(supervised_loader_train)
    print('Training dataset, length:', ipe*batch_size)

    _, supervised_loader_val, supervised_sampler_val = make_PlantCLEF2022(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            supervision=True,
            drop_last=False)
    
    ipe_val = len(supervised_loader_val)
    print('Val dataset, length:', ipe_val*batch_size)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    mixup_fn = None 
    mixup_active = mixup > 0 or cutmix > 0. # or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, label_smoothing=0.1, num_classes=nb_classes) # TODO: use inside the training loop []

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    '''
    # TODO (Fine-tuning parameters) according to ViT Paper:
         _______________________________________________
        |        config          |          value      |
        |------------------------|----------------------      
        | optimizer              |      AdamW          | [x]
        | base learning rate     |      1e-3           | [x]
        | weight decay           |      0.05           | [x]  
        | optimizer momentum     | β1 , β2 =0.9, 0.999 | [x]    
        | layer-wise lr decay    |      0.75           | [x]
        | batch size             |      1024           | [x]
        | learning rate schedule |  cosine decay       | [x]
        | warmup epochs          |      5              | [x]
        | training epochs        | 100 (B), 50 (L/H)   | [x]
        | augmentation           | RandAug (9, 0.5)    | [x]
        | label smoothing        |      0.1            | [x]
        | mixup                  |      0.8            | [x]
        | cutmix                 |      1.0            | [x]
        | drop path [30]         |  0.1 (B/L) 0.2 (H)  | [x]
        -----------------------------------------------
    '''    

    '''
         
        TODO(s):
        (1) - Map overlapping code with vit mae [x]
        (2) - Avg pool the output embeddings and feed them into a mlp with the same config as in MAE []
        (3) - Look up for which encoder was used to extract features in the ablation studies []

        OBS:  lr = base lr×batchsize / 256. (learning rate formula).
    '''
    # -- # -- # -- #
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder, static_graph=True) # TODO: verify what does static_graph do[]

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0

    # TODO:
    # (1) - Make checkpoint a requirement[]
    # (2) - Iterate optmizers when resuming training[]
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
        #for _ in range(start_epoch*ipe):
        #    scheduler.step()
        #    wd_scheduler.step()
        #    next(momentum_scheduler)
        #    mask_collator.step()


    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    a = 0 

    target_encoder = add_classification_head(target_encoder, nb_classes=nb_classes, drop_path=0.2, device=device)
    target_encoder = DistributedDataParallel(target_encoder)

    # TODO:
    # (1) - Create supervised sampler[x]
    # (2) - Send both label and images to device[x]
    # (3) - Apply mixup[x] 
    # (4) - 
    # (-) - Learning Rate Scheduling. 
    # (-) - 
    # (-) - Gradient Clipping. 

    '''
        In distributed mode, calling the set_epoch() method 
        at the beginning of each epoch before creating the DataLoader iterator 
        is necessary to make shuffling work properly across multiple epochs. 
        Otherwise, the same ordering will be always used.
    '''

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        supervised_loader_train.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()


#        csv_logger = CSVLogger(log_file,
#                            ('%d', 'epoch'),
#                            ('%d', 'itr'),
#                            ('%.5f', 'train loss'),
#                            ('%.5f', 'test loss'),
#                            ('%.3f', 'Train - Acc@1'),
#                            ('%.3f', 'Train - Acc@5'),
#                            ('%.3f', 'Test - Acc@1'),
#                            ('%.3f', 'Test - Acc@5'),
#                            ('%d', 'time (ms)'))



        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)        

    max_accuracy = 0.0
    for itr, (udata, masks_enc, masks_pred) in enumerate(supervised_loader_train):
        a +=1 
        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(device, non_blocking=True)
            targets = udata[1].to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets = mixup_fn(imgs, targets)
            return (samples, targets)
        imgs, targets = load_imgs()

        def train_step():
                        
            def loss_fn(h, targets):
                loss = criterion(h, targets)
                loss = AllReduce.apply(loss)
                return loss
                        
            # Step 1. Forward
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                h = target_encoder.forward(imgs)
                loss = loss_fn(h, targets)

            #  Step 2. Backward & step
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            grad_stats = grad_logger(target_encoder.named_parameters())
            optimizer.zero_grad()
            return (float(loss), _new_lr, _new_wd, grad_stats)                
        (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
        loss_meter.update(loss)
        time_meter.update(etime)

        # -- Logging
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

        assert not np.isnan(loss), 'loss is nan'
        print('Loss:', loss)        
        if a == 4:
            break

if __name__ == "__main__":
    main()
    exit(0)

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
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

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
