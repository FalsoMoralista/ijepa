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
    load_FT_checkpoint,
    init_model,
    init_opt,
    init_FT_opt
    )
from src.transforms import make_transforms
import time

# --BROUGHT fRoM MAE
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

# --
log_timings = True
log_freq = 200
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

    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    resume_epoch = args['data']['resume_epoch']

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
        load_path = '/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/' + 'IN1K-vit.h.14-900e.pth.tar'  #os.path.join(folder, r_file) if r_file is not None else latest_path
    
    if resume_epoch > 0:
        r_file = 'jepa-ep{}.pth.tar'.format(resume_epoch + 1)
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'Train loss'),
                           ('%.5f', 'Test loss'),
                           ('%.3f', 'Test - Acc@1'),
                           ('%.3f', 'Test - Acc@5'),
                           ('%d', 'Test time (ms)'),
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

    training_transform = make_transforms( 
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        supervised=True,
        validation=False,
        normalization=((0.436, 0.444, 0.330),
                (0.203, 0.199, 0.195)), # PlantCLEF normalization
        color_jitter=color_jitter)
    
    val_transform = make_transforms( 
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        supervised=True,
        validation=True,
        normalization=((0.436, 0.444, 0.330),
                (0.203, 0.199, 0.195)), # PlantCLEF normalization
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, supervised_loader_train, supervised_sampler_train = make_PlantCLEF2022(
            transform=training_transform,
            batch_size=batch_size,
            collator= None,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(supervised_loader_train)
    print('Training dataset, length:', ipe*batch_size)

    # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. 
    # This will slightly alter validation results as extra duplicate entries are added to achieve
    # equal num of samples per-process.'
    _, supervised_loader_val, supervised_sampler_val = make_PlantCLEF2022(
            transform=val_transform,
            batch_size=batch_size,
            collator= None,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    
    ipe_val = len(supervised_loader_val)

    # The val dataset has length of 546048, which alongside with a batch_size of 96 * 4 gpus = 1422 batches per gpu so distributed testing
    # under these circumstances should be fine. 
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
        mixup_fn = Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, label_smoothing=0.1, num_classes=nb_classes)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # -- # -- # -- #
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    # -- Load ImageNet weights
    if resume_epoch == 0:    
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
    
    def save_checkpoint(epoch):
        save_dict = {
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

    target_encoder = target_encoder.module # Unwrap from DDP    
    for p in target_encoder.parameters():
        p.requires_grad = True

    target_encoder = add_classification_head(target_encoder, nb_classes=nb_classes, drop_path=drop_path, device=device)

    # -- Override previously loaded optimization configs.
    optimizer, scaler, scheduler, wd_scheduler = init_FT_opt(
        encoder=target_encoder,
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
    
    target_encoder = DistributedDataParallel(target_encoder, static_graph=True) # Static Graph: the set of used and unused parameters will not change during the whole training loop.
    
    if resume_epoch != 0:
        target_encoder, optimizer, scaler, start_epoch = load_FT_checkpoint(
            device=device,
            r_path=load_path,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(resume_epoch*ipe):
            scheduler.step() 
            wd_scheduler.step()

    logger.info(target_encoder)

    # -- Remove from device once they are required for loading pretrained parameters only
    del encoder
    del predictor
    
    accum_iter = 4
    start_epoch = resume_epoch

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        
        logger.info('Epoch %d' % (epoch + 1))
        # In distributed mode, calling the set_epoch() method 
        # at the beginning of each epoch before creating the DataLoader iterator
        # is necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.        
        supervised_sampler_train.set_epoch(epoch) # -- update distributed-data-loader epoch
        
        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        target_encoder.train(True)
        for itr, (sample, target) in enumerate(supervised_loader_train):
            def load_imgs():
                samples = sample.to(device, non_blocking=True)
                targets = target.to(device, non_blocking=True)

                if mixup_fn is not None:
                    samples, targets = mixup_fn(samples, targets)
                return (samples, targets)
            imgs, targets = load_imgs()

            def train_step():    
                _new_lr = scheduler.step() 
                _new_wd = wd_scheduler.step()
                         
                def loss_fn(h, targets):
                    loss = criterion(h, targets)
                    loss = AllReduce.apply(loss)
                    # TODO: verify below.
                    # It is not necessary to use another allreduce to sum all loss. 
                    # Additional allreduce might have considerable negative impact on training speed.
                    # See: https://discuss.pytorch.org/t/distributeddataparallel-loss-compute-and-backpropogation/47205/4                    
                    return loss
                            
                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = target_encoder(imgs)
                    loss = loss_fn(h, targets)

                #  Step 2. Backward & step
                if use_bfloat16:
                    loss /= accum_iter
                    scaler(loss, optimizer, clip_grad=None,
                                parameters=target_encoder.parameters(), create_graph=False,
                                update_grad=(itr + 1) % accum_iter == 0)
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(target_encoder.named_parameters())
                if (itr + 1) % accum_iter == 0:
                    optimizer.zero_grad()

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d/%5d] train_loss: %.3f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'

                                % (epoch + 1, itr, ipe,
                                    loss_meter.avg,
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

        testAcc1 = AverageMeter()
        testAcc5 = AverageMeter()
        test_loss = AverageMeter()

        # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number
        # will slightly alter validation results as extra duplicate entries are added to achieve equal 
        # num of samples per-process.
        @torch.no_grad()
        def evaluate(unused_parameters=None):
            crossentropy = torch.nn.CrossEntropyLoss()

            target_encoder.eval()              
            supervised_sampler_val.set_epoch(epoch) # -- Enable shuffling to reduce monitor bias
            
            for cnt, (samples, targets) in enumerate(supervised_loader_val):
                images = samples.to(device, non_blocking=True)
                labels = targets.to(device, non_blocking=True)
                                 
                with torch.cuda.amp.autocast():
                    output = target_encoder.forward(images)
                    loss = crossentropy(output, labels)
                
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                testAcc1.update(acc1)
                testAcc5.update(acc5)
                test_loss.update(loss)
        
        vtime = gpu_timer(evaluate)
        
        #logger.info('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Test loss {losses.avg:.3f}'.format(top1=testAcc1, top5=testAcc5, losses=test_loss))
        
        # -- Logging
        def log_test():
            csv_logger.log(epoch + 1, test_loss.val, testAcc1.avg, testAcc5.avg, vtime)
            if (itr % log_freq == 0) or np.isnan(test_loss) or np.isinf(test_loss): # TODO: fix TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

                logger.info('[%d] test_loss: %.3f '
                            ' - test_acc1 - [%.3f], test_acc5 - [%.3f]'
                            '[mem: %.2e] '
                            '(%.1f ms)'

                            % (epoch + 1,
                                test_loss.avg,
                                testAcc1.avg, testAcc5.avg,
                                torch.cuda.max_memory_allocated() / 1024.**2,
                                vtime))
        #log_test()

        # -- Save Checkpoint after every epoch
        logger.info('avg. train_loss %.3f' % loss_meter.avg)
        logger.info('avg. test_loss %.3f avg. Accuracy@1 %.3f - avg. Accuracy@5 %.3f' % (test_loss.avg, testAcc1.avg, testAcc5.avg))
        save_checkpoint(epoch+1)
        assert not np.isnan(loss), 'loss is nan'
        print('Loss:', loss)        

if __name__ == "__main__":
    main()
