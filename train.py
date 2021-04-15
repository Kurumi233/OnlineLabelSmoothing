import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision

from models.model import BaseModel
from models.FocalLoss import FocalLoss
from models.LabelSmoothing import LabelSmoothing
from models.OLS import OnlineLabelSmoothing
from ImageNetLoad import ImageNet
from utils import MultiLossAverageMeter, AverageMeter, accuracy
from plot_result import plot_result

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='res50', type=str)
parser.add_argument('--savepath', default='./Single_gpu', type=str)
parser.add_argument('--loss', default=['ce'], nargs='+', type=str)
parser.add_argument('--loss_w', default=[1.], nargs='+', type=float)
parser.add_argument('--smoothing', default=0.1, type=float)
parser.add_argument('-c', '--num_classes', default=1000, type=int)
parser.add_argument('-p', '--pool_type', default='avg', type=str)
parser.add_argument('--metric', default='linear', type=str)
parser.add_argument('--down', default=0, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('-s', '--scheduler', default='step', type=str)
parser.add_argument('-r', '--resume', default=None, type=str)
parser.add_argument('--lr_step', default=30, type=int)
parser.add_argument('--warm', default=5, type=int)
parser.add_argument('--print_step', default=500, type=int)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--total_epoch', default=250, type=int)
parser.add_argument('-bs', '--batch_size', default=256, type=int)
parser.add_argument('-nw', '--num_workers', default=20, type=int)
parser.add_argument('--multi-gpus', default=0, type=int)
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--pretrained', default=0, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--sync_bn', default=False, action='store_true')
parser.add_argument('--amp', default=False, action='store_true')

args = parser.parse_args()
print('local_rank:', args.local_rank)

ce_based_loss = ['ce', 'ls', 'fl', 'ols']


def loss_func(x, target, feat=None, training=False):
    loss_dict = {}
    loss_value = 0.
    for l, w in zip(args.loss, args.loss_w):
        if training:
            criterion[l].train()
        else:
            criterion[l].eval()
        if l in ce_based_loss:
            loss = w * criterion[l](x, target)
        loss_value += loss
        loss_dict[l] = loss.detach().cpu().item()

    return loss_dict, loss_value


def train(epoch):
    model.train()

    loss_meter = MultiLossAverageMeter(args.loss)
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    t1 = time.time()
    s1 = time.time()
    for idx, (data, labels) in enumerate(trainloader):
        if multi_gpus:
            data, labels = data.cuda(non_blocking=True), labels.long().cuda(non_blocking=True)
        else:
            data, labels = data.to(device), labels.long().to(device)

        optimizer.zero_grad()
        
        # AMP
        if args.amp:
            with torch.cuda.amp.autocast():
                out, feat = model(data)
                loss_dict, loss = loss_func(out, labels, feat, training=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out, feat = model(data)
            loss_dict, loss = loss_func(out, labels, feat, training=True)

            loss.backward()
            optimizer.step()

        loss_meter.update(loss_dict, data.size(0))
        acc1, acc5 = accuracy(out, labels, topk=(1, 5))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        if idx % args.print_step == 0:
            s2 = time.time()
            print(f'rank:{args.local_rank} epoch[{epoch:>3}/{args.total_epoch}] idx[{idx:>3}/{len(trainloader)}] loss[{loss_meter}] acc[@1:{top1.avg:.4f} @5:{top5.avg:.4f}] time:{s2 - s1:.2f}s')
            s1 = time.time()

    if args.local_rank == 0:
        print('=' * 30)
    print(f'rank:{args.local_rank} train loss[{loss_meter}] acc[@1:{top1.avg:.4f} @5:{top5.avg:.4f}] time:{time.time() - t1:.2f}s')

    if args.local_rank == 0:
        with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
            f.write('epoch:{} lr:{:.8f} loss[{}] acc[@1:{:.4f} @5:{:.4f}] '.format(epoch, optimizer.param_groups[0]['lr'], loss_meter, top1.avg, top5.avg))


def test(epoch):
    model.eval()

    loss_meter = MultiLossAverageMeter(args.loss)
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    with torch.no_grad():
        for idx, (data, labels) in enumerate(valloader):
            data, labels = data.to(device), labels.long().to(device)
            out = model(data)
            loss_dict, loss = loss_func(out, labels, training=False)

            loss_meter.update(loss_dict, data.size(0))
            acc1, acc5 = accuracy(out, labels, topk=(1, 5))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

        print(f'rank:{args.local_rank} test  loss[{loss_meter}] acc[@1:{top1.avg:.4f} @5:{top5.avg:.4f}]', end=' ')

    global best_acc, best_epoch

    if isinstance(model, nn.parallel.distributed.DistributedDataParallel):
        state = {
            'net': model.module.state_dict(),
            'acc': top1.avg,
            'epoch': epoch,
        }
    else:
        state = {
            'net': model.state_dict(),
            'acc': top1.avg,
            'epoch': epoch,
        }
    
    if 'ols' in args.loss:
        state['ols'] = criterion['ols'].matrix.cpu().data
    
    if top1.avg > best_acc:
        best_acc = top1.avg
        best_epoch = epoch
        torch.save(state, os.path.join(savepath, 'best.pth'))
        print('*')
    else:
        print()

    torch.save(state, os.path.join(savepath, 'last.pth'))

    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('test loss[{}] acc[@1:{:.4f} @5:{:.4f}]\n'.format(loss_meter, top1.avg, top5.avg))


if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0.
    use_gpu = False
    multi_gpus = False
    
    start_epoch = 0
    total = args.total_epoch

    if args.seed is not None:
        print('use random seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = False

    if torch.cuda.is_available():
        use_gpu = True
        cudnn.benchmark = True

    if torch.cuda.device_count() > 1 and args.multi_gpus:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        multi_gpus = True

    # loss
    criterion = {
        'ce': nn.CrossEntropyLoss(),
        'fl': FocalLoss(),
        'ls': LabelSmoothing(smoothing=args.smoothing),
        'ols': OnlineLabelSmoothing(num_classes=args.num_classes, use_gpu=use_gpu)
    }

    # dataloader
    trainset = ImageNet(mode='train')
    valset = ImageNet(mode='val')

    # dataloader
    train_sampler = None
    if multi_gpus:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=args.batch_size,
                             shuffle=(train_sampler is None),
                             sampler=train_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)

    valloader = DataLoader(dataset=valset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=True)

    # model
    model = BaseModel(model_name=args.model_name,
                      num_classes=args.num_classes,
                      pretrained=args.pretrained)
    
    if args.resume:
        state = torch.load(args.resume)
        print('Resume from:{}'.format(args.resume))
        model.load_state_dict(state['net'], strict=False)
        best_acc = state['acc']
        start_epoch = state['epoch'] + 1
        if 'ols' in args.loss:
            criterion['ols'].matrix = state['ols'].cuda()
    
    # sync_bn
    if args.sync_bn and multi_gpus:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Using SyncBatchNorm')
    
    if multi_gpus:
        device = torch.device("cuda", args.local_rank)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        device = ('cuda:%d' % args.local_rank if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    print('Device:', device)

    # optim
    optimizer = torch.optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}],
        weight_decay=args.weight_decay, momentum=args.momentum)

    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'multi':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 225], gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'cos':
        warm_up_step = args.warm
        lambda_ = lambda epoch: (epoch + 1) / warm_up_step if epoch <= warm_up_step else 0.5 * (
                np.cos((epoch - warm_up_step) / (args.total_epoch - warm_up_step) * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)
    else:
        raise ValueError('No such scheduler - {}'.format(args.scheduler))

    # savepath
    loss_str = '_'.join(args.loss)
    if 'ls' in args.loss:
        loss_str += str(args.smoothing)
    savepath = os.path.join(args.savepath, '{}_{}_{}_{}_{}'.format(args.model_name,
                                                                args.pool_type,
                                                                args.metric,
                                                                str(args.down),
                                                                loss_str))
    # AMP
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        print('Using Mixing Accuracy.')
        savepath += '_amp'
    
    if args.sync_bn:
        savepath += '_syncbn'
    
    savepath += args.scheduler

    if args.local_rank == 0:
        print('Init_lr={}, Weight_decay={}, Momentum={}'.format(args.lr, args.weight_decay, args.momentum))
        print('Loss:', args.loss)
        print('Loss_weight:', args.loss_w)
        print('Using {} scheduler'.format(args.scheduler))
        print('Savepath:', savepath)

    os.makedirs(savepath, exist_ok=True)

    if args.local_rank == 0 and args.resume is None:
        with open(os.path.join(savepath, 'setting.txt'), 'w')as f:
            for k, v in vars(args).items():
                f.write('{}:{}\n'.format(k, v))

        f = open(os.path.join(savepath, 'log.txt'), 'w')
        f.close()
    
    start = time.time()
    
    for epoch in range(start_epoch, total):
        train(epoch)
        scheduler.step()
        if args.local_rank == 0:
            test(epoch)
        if 'ols' in args.loss:
            criterion['ols'].update()

    end = time.time()
    if args.local_rank == 0:
        print('total time:{}m{:.2f}s'.format((end - start) // 60, (end - start) % 60))
        print('best_epoch:', best_epoch)
        print('best_acc:', best_acc)
        with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
            f.write('# best_acc:{:.4f}, best_epoch:{}'.format(best_acc, best_epoch))

        plot_result(txt=os.path.join(savepath, 'log.txt'), savepath=savepath)