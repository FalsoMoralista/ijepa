# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
import torch.nn  as nn
import torch.nn.functional as F
from torch import inf 

# from timm.models.layers import trunc_normal_ 
import util.lr_decay as lrd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch

def load_FT_checkpoint(
    device,
    r_path,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return target_encoder, opt, scaler, epoch

class FinetuningModel(nn.Module):
    def __init__(self, pretrained_model, drop_path, nb_classes):
        super(FinetuningModel, self).__init__()        
        self.pretrained_model = pretrained_model
        
        self.drop_path = drop_path
        self.nb_classes = nb_classes
        
        self.pretrained_model.drop_path = 0.2  # Does it change anything after the model has been loaded? TODO: REMEMBER THIS WAS ADDED
        self.pretrained_model.drop_rate = 0.25
        
        self.n_intermediate_outputs = 4
        
        #self.pretrained_model.layer_dropout = self.drop_path 

        self.average_pool = nn.AvgPool1d((self.pretrained_model.patch_embed.num_patches), stride=1)

        self.norm = nn.LayerNorm(self.pretrained_model.embed_dim) # TODO: REMOVE

        self.head_drop = nn.Dropout(drop_path)

        #self.mlp_head = nn.Linear(self.n_intermediate_outputs * self.pretrained_model.embed_dim,
        #                            self.nb_classes)
           
        self.mlp_head = nn.Linear(self.pretrained_model.embed_dim,
                                    self.nb_classes)   

    '''
        " Because our I-JEPA implementation uses Vision Transformer architectures without a [cls] token, 
        we adapt the default VISSL evaluation recipe to utilize the average-pooled patch representation
        instead of the [cls] token. 
        We therefore report the best linear evaluation number among the following representations: 
        1) the average-pooled patch representation of the last layer,
        2) the concatenation of the last 4 layers of the average-pooled patch representations."    
    '''
    def get_n_intermediate_outputs(self, n , x):

        # -- patchify x
        x = self.pretrained_model.patch_embed(x) # -> (B, 256, 1280)

        # -- add positional embedding to x 
        pos_embed = self.pretrained_model.interpolate_pos_encoding(x, self.pretrained_model.pos_embed) # See vision_transformer.py @Line 410 for more info.
        x = x + pos_embed
        
        # Extract the representation (B, 256, 1280) from the last 4 layers
        # averaging them individually -> 4x (B, 1, 1280), then
        # concatenate into a single representation (B, 5120).
        outputs = []
        n_blocks = len(self.pretrained_model.blocks) - 1            
        layers = [(n_blocks - i) for i in reversed(range(n))]
        # -- 1. fwd prop
        for b, blk in enumerate(self.pretrained_model.blocks):
            x = blk(x)
            # -- 2. Patch-wise averaging and normalization.
            if b in layers:
                h = self.average_pool(x.transpose(1, 2)).transpose(1, 2)
                h = h.squeeze(1) # adjust
                h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim    
                outputs.append(h)
                        
        # -- 3. Concatenation
        output = torch.cat(outputs, dim=-1)
        return output

    def forward_with_intermediate_outputs(self, x):

        x = self.get_n_intermediate_outputs(self.n_intermediate_outputs, x)
        
        x = self.head_drop(x) # As performed in timm.models
        
        x = self.mlp_head(x)
        return x


    def forward(self, x):

        x = self.pretrained_model(x)

        x = self.average_pool(x.transpose(1, 2)).transpose(1, 2) # conduct average pool like in paper

        x = x.squeeze(1)

        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        
        x = self.head_drop(x) # As in in timm.models
        
        x = self.mlp_head(x)
        
        return x

    
def add_classification_head(pretrained_model, drop_path, nb_classes, device):
    model = FinetuningModel(pretrained_model, drop_path, nb_classes)
    
    # manually initialize fc layer (borrowed from MAE)
    trunc_normal_(model.mlp_head.weight, std=2e-5) 
    
    model.to(device)
    return model         
        

# Borrowed from MAE.
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    # logger.info(encoder)
    return encoder, predictor

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0 
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_FT_opt(
    encoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0 
        }
    ]

    # build optimizer with layer-wise lr decay (lrd)
    #param_groups = lrd.param_groups_lrd(encoder.pretrained_model, wd,
    #    no_weight_decay_list={'pos_embed', 'cls_token', 'dist_token'}, # gathered here https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L573
    #    layer_decay=0.75
    #)
    
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = NativeScalerWithGradNormCount() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler 