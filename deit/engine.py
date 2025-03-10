# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
import utils

from torch import nn

import io, os

def get_sparsity_loss(model, optimizer):
    sparsity_loss_attn, sparsity_loss_mlp = 0, 0
    for i in range(model.layers):
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = optimizer.state[p]
                #print("group ipdate\n\n")
                #print(param_state['name'])
                if ('alpha' in param_state['name']) :
                    #print("visual",i,param_state['name'])
                    if 'attn' in param_state['name']:
                        sparsity_loss_attn += torch.sum(torch.abs(getattr(model.blocks, str(i)).attn.alpha-param_state['gamma_buffer'].cuda())**2)
                    if 'mlp' in param_state['name']:
                        sparsity_loss_mlp += torch.sum(torch.abs(getattr(model.blocks, str(i)).mlp.alpha-param_state['gamma_buffer'].cuda())**2)
    return sparsity_loss_attn, sparsity_loss_mlp


def compress_model(model,num_heads=6):
    '''
        compress model with alpha, after compress should load model by other ways (not model.load_stat_dict)
    '''
    # attn
    import pdb
    # pdb.set_trace()
    #num_heads = num_heads
    head_dim = int(model.embed_dim / num_heads)

    for i in range(len(model.blocks)):
        getattr(model.blocks,str(i)).attn._qkv_same_embed_dim = False
        in_features    = getattr(model.blocks,str(i)).attn.qkv.weight.shape[-1]  # 768
        out_features   = getattr(model.blocks,str(i)).attn.proj.weight.shape[0]  # 768
        pj_in_features = getattr(model.blocks,str(i)).attn.proj.weight.shape[1]  # 768
        if hasattr(getattr(model.blocks,str(i)).attn,"alpha"):
            alpha = torch.ones_like(getattr(model.blocks,str(i)).attn.alpha.squeeze())
            alpha.copy_(getattr(model.blocks,str(i)).attn.alpha.squeeze())
            #alpha[alpha!=0]=1
        else:
            alpha = getattr(model.blocks,str(i)).attn.score.squeeze()
            alpha[alpha>getattr(model.blocks,str(i)).attn.threshold]=1
        #            64
        # alpha_qk = alpha[ : head_dim].repeat(num_heads) # 1 * 64 * 12
        # alpha_v  = alpha[2 * head_dim: ].repeat(num_heads) # 1 * n

        hidden_features_emb = torch.count_nonzero(alpha==1)
        # if hidden_features_emb%8!=0:
        #     hidden_features_t = (hidden_features_emb//8 + 1 ) * 8
        #     save_alpha = hidden_features_t - hidden_features_emb
        #     non_one_indices = torch.where(alpha_qk != 1)[0]
        #     #print(alpha[alpha!=1][:save_alpha])
        #     alpha_qk[non_one_indices[:save_alpha]] = 1
        alpha_qk = alpha[:head_dim].repeat(num_heads) # 1 * 64 * 12
        alpha_v  = alpha[head_dim:].repeat(num_heads) # 1 * n

        hidden_features_qk = torch.count_nonzero(alpha_qk==1)
        hidden_features_v = torch.count_nonzero(alpha_v==1)


        if hidden_features_v!=0:
        # resolve attention
            getattr(model.blocks,str(i)).attn.w_q = nn.Linear(in_features, hidden_features_qk)
            getattr(model.blocks,str(i)).attn.w_k = nn.Linear(in_features, hidden_features_qk)
            getattr(model.blocks,str(i)).attn.w_v = nn.Linear(in_features, hidden_features_v)

            getattr(model.blocks,str(i)).attn.w_q.weight.data = getattr(model.blocks,str(i)).attn.qkv.weight.data[: pj_in_features, :][alpha_qk==1, :]
            getattr(model.blocks,str(i)).attn.w_q.bias.data   = getattr(model.blocks,str(i)).attn.qkv.bias.data[: pj_in_features][alpha_qk==1]
            getattr(model.blocks,str(i)).attn.w_k.weight.data = getattr(model.blocks,str(i)).attn.qkv.weight.data[pj_in_features : 2 * pj_in_features, :][alpha_qk==1, :]
            getattr(model.blocks,str(i)).attn.w_k.bias.data   = getattr(model.blocks,str(i)).attn.qkv.bias.data[pj_in_features : 2 * pj_in_features][alpha_qk==1]
            getattr(model.blocks,str(i)).attn.w_v.weight.data = getattr(model.blocks,str(i)).attn.qkv.weight.data[2 * pj_in_features:, :][alpha_v==1, :]
            getattr(model.blocks,str(i)).attn.w_v.bias.data   = getattr(model.blocks,str(i)).attn.qkv.bias.data[2 * pj_in_features: ][alpha_v==1]

            temp = nn.Linear(pj_in_features, out_features)
            temp.weight.data = getattr(model.blocks, str(i)).attn.proj.weight.data
            temp.bias.data   = getattr(model.blocks, str(i)).attn.proj.bias.data
            getattr(model.blocks,str(i)).attn.proj = nn.Linear(hidden_features_v, out_features)
            getattr(model.blocks,str(i)).attn.proj.weight.data = temp.weight.data[:, alpha_v==1]
            getattr(model.blocks,str(i)).attn.proj.bias.data = temp.bias.data
        
            temp.weight.data = torch.tensor([])
            temp.bias.data = torch.tensor([])

            module = getattr(model.blocks,str(i)).attn
            if hasattr(module, 'qkv'):
                delattr(module, 'qkv')
            if hasattr(module, 'threshold'):
                delattr(module, 'threshold')
            if hasattr(module, 'alpha'):
                delattr(module, 'alpha')
        else:
            module = getattr(model.blocks,str(i))
            if hasattr(module, 'attn'):
                delattr(module, 'attn')
            if hasattr(module, 'norm1'):
                delattr(module, 'norm1')
        # pdb.set_trace()

    # mlp
    for i in range(len(model.blocks)):

        in_features    = getattr(model.blocks,str(i)).mlp.fc1.weight.shape[-1]
        out_features   = getattr(model.blocks,str(i)).mlp.fc2.weight.shape[0]
        pj_in_features = getattr(model.blocks,str(i)).mlp.fc1.weight.shape[0]

        if hasattr(getattr(model.blocks,str(i)).mlp,"alpha"):
            alpha = torch.ones_like(getattr(model.blocks,str(i)).mlp.alpha.squeeze())
            alpha.copy_(getattr(model.blocks,str(i)).mlp.alpha.squeeze())
            #alpha[alpha!=0]=1
        else:
            alpha = getattr(model.blocks,str(i)).mlp.score.squeeze()
            alpha[alpha>getattr(model.blocks,str(i)).mlp.threshold]=1


        #alpha[alpha < 0.1] = 0.
        hidden_features = torch.count_nonzero(alpha==1)
        
        temp = nn.Linear(in_features, pj_in_features, bias=True)
        temp.weight.data = getattr(model.blocks,str(i)).mlp.fc1.weight.data
        temp.bias.data   = getattr(model.blocks,str(i)).mlp.fc1.bias.data

        getattr(model.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
        getattr(model.blocks, str(i)).mlp.fc1.weight.data = temp.weight.data[alpha==1, :]
        getattr(model.blocks, str(i)).mlp.fc1.bias.data   = temp.bias.data[alpha==1]

        temp = nn.Linear(pj_in_features, out_features,bias=True)
        temp.weight.data = getattr(model.blocks,str(i)).mlp.fc2.weight.data
        temp.bias.data   = getattr(model.blocks,str(i)).mlp.fc2.bias.data
        getattr(model.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
        getattr(model.blocks, str(i)).mlp.fc2.weight.data = temp.weight.data[:, alpha==1]
        getattr(model.blocks, str(i)).mlp.fc2.bias.data   = temp.bias.data
        
        module = getattr(model.blocks,str(i)).mlp
        if hasattr(module, 'threshold'):
            delattr(module, 'threshold')
        if hasattr(module, 'alpha'):
            delattr(module, 'alpha')
        temp.weight.data = torch.tensor([])
        temp.bias.data   = torch.tensor([])
        # import pdb
        # pdb.set_trace()
    print("compress model success !")

def set_alpha_sparsity(model,sparsity):
    layers=model.layers
    alpha_attn = torch.stack([getattr(model.blocks, str(i)).attn.alpha for i in range(layers)])
    alpha_mlp = torch.stack([getattr(model.blocks, str(i)).mlp.alpha for i in range(layers)])

    alpha = alpha_attn.view(-1)
    sorted_alpha_value, indices = torch.sort(alpha, descending=False)
    compression_weight = torch.ones_like(indices)
    #compression_weight[indices < alpha_grad_attn.numel()] = 9234/769
    threshold_attn = sorted_alpha_value[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*sparsity))]
    
    alpha = alpha_mlp.view(-1)
    sorted_alpha_value, indices = torch.sort(alpha, descending=False)
    compression_weight = torch.ones_like(indices)
    #compression_weight[indices < alpha_grad_attn.numel()] = 9234/769
    threshold_mlp = sorted_alpha_value[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*sparsity))]
    

    def update(module, value,threshold):
        mask = (value >= threshold) 
        module.data.copy_(mask)

    for i in range(layers):
        update(getattr(model.blocks, str(i)).attn.alpha, alpha_attn[i],threshold_attn)
        update(getattr(model.blocks, str(i)).mlp.alpha, alpha_mlp[i],threshold_mlp)

def prune_if_compressed(model , state):
    #print(state.keys())
    state_dict = state['model']
    print(state_dict.keys())
    for i in range(12):
        getattr(model.blocks,str(i)).attn._qkv_same_embed_dim = False
        if getattr(model.blocks, str(i)).mlp.fc1.weight.shape != state_dict[f'blocks.{i}.mlp.fc1.weight'].shape:
            
            getattr(model.blocks, str(i)).mlp.fc1 = nn.Linear(*state_dict[f'blocks.{i}.mlp.fc1.weight'].shape[::-1])
            getattr(model.blocks, str(i)).mlp.fc2 = nn.Linear(*state_dict[f'blocks.{i}.mlp.fc2.weight'].shape[::-1])
            # delattr(getattr(getattr(model.layers,str(i)).blocks,str(j)).mlp, 'score')
            # delattr(getattr(getattr(model.layers,str(i)).blocks,str(j)).mlp, 'threshold')
        if f'blocks.{i}.attn.w_q.weight' in state_dict:
            getattr(model.blocks, str(i)).attn.w_q  = nn.Linear(*state_dict[f'blocks.{i}.attn.w_q.weight'].shape[::-1])
            getattr(model.blocks, str(i)).attn.w_k  = nn.Linear(*state_dict[f'blocks.{i}.attn.w_k.weight'].shape[::-1])
            getattr(model.blocks, str(i)).attn.w_v  = nn.Linear(*state_dict[f'blocks.{i}.attn.w_v.weight'].shape[::-1])
            getattr(model.blocks, str(i)).attn.proj = nn.Linear(*state_dict[f'blocks.{i}.attn.proj.weight'].shape[::-1])
            delattr(getattr(model.blocks, str(i)).attn,'qkv')
        else:
            delattr(getattr(model.blocks, str(i)),'attn')
            delattr(getattr(model.blocks, str(i)),'norm1')

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None,
                    search=False, update_alpha=True,writer=None,state_dict=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if search:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_mlp', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        layers = model.module.layers

    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    print_freq = 10

    len_data_loader = len(data_loader)
    total_steps = len_data_loader*args.epochs if not search else len_data_loader*args.epochs_search

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if i>=10:
        #     break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)
        step=len_data_loader*epoch+i
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()    
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if search and ((i % args.interval == 0 or step == total_steps - 1)):
            sparsity_attn,sparsity_mlp,sparsity_total,z_max_attn_col,z_max_mlp_col=optimizer.print_sparsity_all()
            sparsity_loss_attn, sparsity_loss_mlp = get_sparsity_loss(model.module, optimizer)
            metric_logger.update(loss_ce=loss.item()) 
            metric_logger.update(loss_sp_attn=args.w_sp_attn * sparsity_loss_attn.item()) 
            metric_logger.update(loss_sp_mlp=args.w_sp_mlp * sparsity_loss_mlp.item()) 
            loss += args.w_sp_attn * sparsity_loss_attn + args.w_sp_mlp * sparsity_loss_mlp
            step = epoch*len_data_loader+i
            writer.add_scalar('Mask/Sparsity_attn', sparsity_attn, i+len_data_loader*epoch)
            writer.add_scalar('Mask/Sparsity_mlp', sparsity_mlp, i+len_data_loader*epoch)
            writer.add_scalar('Mask/Sparsity_total', sparsity_total, i+len_data_loader*epoch)
            writer.add_scalar('Mask/Zbuffer_attn', z_max_attn_col, i+len_data_loader*epoch)
            writer.add_scalar('Mask/Zbuffer_mlp', z_max_mlp_col, i+len_data_loader*epoch)
            writer.add_scalar('Train/Loss', loss.item(), i)
       
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
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


        
    if client is not None:
        with io.BytesIO(client.get(os.path.join('s3://BucketName/ProjectName', url_or_filename), enable_cache=True)) as f:
            checkpoint = torch.load(f, map_location='cpu')
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']

    for i in range(model.layers):
        # mlp
        if getattr(model.blocks, str(i)).mlp.fc1.weight.shape != state_dict['blocks.'+str(i)+'.mlp.fc1.weight'].shape:
            del getattr(model.blocks, str(i)).mlp.fc1
            getattr(model.blocks, str(i)).mlp.fc1 = nn.Linear(*state_dict['blocks.'+str(i)+'.mlp.fc1.weight'].shape[::-1])
            del getattr(model.blocks, str(i)).mlp.fc2
            getattr(model.blocks, str(i)).mlp.fc2 = nn.Linear(*state_dict['blocks.'+str(i)+'.mlp.fc2.weight'].shape[::-1])

        # attn
        if getattr(model.blocks, str(i)).attn.qkv.weight.shape != state_dict['blocks.'+str(i)+'.attn.qkv.weight'].shape:
            del getattr(model.blocks, str(i)).attn.qkv
            getattr(model.blocks, str(i)).attn.qkv = nn.Linear(*state_dict['blocks.'+str(i)+'.attn.qkv.weight'].shape[::-1])
            del getattr(model.blocks, str(i)).attn.proj
            getattr(model.blocks, str(i)).attn.proj = nn.Linear(*state_dict['blocks.'+str(i)+'.attn.proj.weight'].shape[::-1])

    torch.cuda.empty_cache()
    model.load_state_dict(state_dict, strict=False)