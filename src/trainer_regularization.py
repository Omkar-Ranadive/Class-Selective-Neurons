'''
Credits: Adapted from https://github.com/pytorch/examples/tree/151944ecaf9ba2c8288ee550143ae7ffdaa90a80/imagenet 
'''

import argparse
import os
import random
import shutil
import time
import warnings
import logging 
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from constants import IMGNET_PATH, DATA_PATH
from class_selectivity_reg import get_class_selectivity, get_selectivity_grad
import utils



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset', default=IMGNET_PATH)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--load_dir", default='', type=str, help='Name of folder to load checkpoints from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--exp_name', required=True, type=str)
parser.add_argument('--inner_save', default=None, type=int, 
                    help="Save within checkpoints")
parser.add_argument("--sel_count", default=None, type=int, required=True, help="Number of times selectivity is calculated during each epoch")
parser.add_argument('--save_batch_targets', action='store_true', help='Save batch target labels for each epoch')
parser.add_argument("--use_ws", action='store_true', help='If true, weighted sampler is used')
parser.add_argument("--alpha", required=True, type=float)

best_acc1 = 0
best_acc5 = 0 
train_acc1 = 0 
train_acc5 = 0 

 
def main():
    args = parser.parse_args()

    global EXP_DIR
    global LOAD_DIR 
    global logger 

    EXP_DIR = DATA_PATH / args.exp_name 
    os.makedirs(EXP_DIR, exist_ok=True)

    if args.load_dir: 
        LOAD_DIR = DATA_PATH / args.load_dir
    else: 
        LOAD_DIR = DATA_PATH 

    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='w')
    logger = logging.getLogger()

    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Training epochs: {args.epochs}')
    logger.info(f'Learning Rate: {args.lr}')
    logger.info(f'Model architecture: {args.arch}')
    logger.info(f'Alpha: {args.alpha}')
    logger.info(f'Start Epoch: {args.resume}')
    logger.info(f'Selectivity calculated {args.sel_count} per epoch')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc5 
    global train_acc1 
    global train_acc5 


    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            print("Running distributed data parallelism")
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("Running normal data parallelism")
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(LOAD_DIR / args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(LOAD_DIR / args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(LOAD_DIR / args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # Make sure the remaining keys exists 
            if 'best_acc5' in checkpoint: 
                best_acc5 = checkpoint['best_acc5']
                train_acc5 = checkpoint['train_acc5']
                train_acc1 = checkpoint['train_acc1']
            else: 
                best_acc5, train_acc5, train_acc1 = 0, 0, 0 

            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
                best_acc5, train_acc5, train_acc1 = best_acc5.to(args.gpu), train_acc5.to(args.gpu), train_acc1.to(args.gpu)

            # Load the state dict 
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(LOAD_DIR / args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    elif args.use_ws: 
        weight_cycle = list(map(float, input("Enter weights: ").split()))
        cycle_switch = int(input("Enter num of examples after which weight gets cycled: "))
        cycle_counter = -1 
        weights = []

        logger.info('Using weighted sampler')
        logger.info(f'Weights used: {weight_cycle}')
        logger.info(f'Cycle switch used: {cycle_switch}')

        for i in range(len(train_dataset)): 
            if i % cycle_switch == 0 and cycle_counter < len(weight_cycle)-1: 
                cycle_counter += 1 
            
            weights.append(weight_cycle[cycle_counter])

        train_sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=False)

    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    if not args.resume: 
        # Save initialized weights 
        acc1, acc5 = validate(val_loader, model, criterion, args)
        save_checkpoint({
            'epoch': 0,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': acc1,
            'best_acc5': acc5, 
            'val_acc1': acc1, 
            'val_acc5': acc5, 
            'train_acc1': train_acc1,
            'train_acc5': train_acc5,
            'optimizer' : optimizer.state_dict(),
        }, False, filename='checkpoint_e{}.pth.tar'.format(0))

    # loader_cp = utils.load_imagenet_data(dir=IMGNET_PATH / 'val', batch_size=32, num_workers=args.workers)



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc1, train_acc5 = train(train_loader, val_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        # train_acc1 = max(train_acc1, acc_t1)
        # train_acc5 = max(train_acc5, acc_t5)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5, 
                'val_acc1': acc1, 
                'val_acc5': acc5, 
                'train_acc1': train_acc1,
                'train_acc5': train_acc5,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename='checkpoint_e{}.pth.tar'.format(epoch+1))


def train(train_loader, val_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    num_batches = len(train_loader) 
    batch_targets = []

    if args.inner_save: 
        # Calculate the batch after which to save 
        save_every = num_batches // args.inner_save 
        print("Num batches {}  Total saves per checkpoint {} Save Every {}".format(
            num_batches, args.inner_save, save_every))

    if args.sel_count: 
        num_cal = num_batches // args.sel_count 
        print(f"Num batches {num_batches}, Num of calculations: {args.sel_count}, Calculate every {num_cal}")

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    epsilon = 1e-6

    
    end = time.time()

  

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # # get class selectivity
        # cs_dict_path = LOAD_DIR / 'cs_dict_val_cp{}'.format(args.start_cp)
        # class_selectivity = utils.load_file(cs_dict_path)

        # compute output
        # output = model(images)
        # if i % num_cal == 0:
        #     class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=val_loader) 

        # output = model(images)

        resnet_layers = nn.Sequential(*list(model.module.children()))
        output = torch.clone(images).to('cuda')

        class_activations = {
        4: {},
        5: {},
        6: {},
        7: {}
        }

        class_selectivity = {
            4: {},
            5: {},
            6: {},
            7: {}
        }

        for index, layer in enumerate(resnet_layers):
            output, class_activations = get_selectivity_grad(index, layer, class_activations, output, target) 
       
        # Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... } 
        for layer_k, layer_v in class_activations.items():
            # for class_k, class_v in class_activations[layer_k].items():
            # For a layer, the number of bottleneck layers will be the same 
            # So, just choose any class to get the index of bottleneck layers 
            random_key = target[0].item()

            for bottleneck_k, bottleneck_v in class_activations[layer_k][random_key].items():
                for ci, class_k in enumerate(sorted(class_activations[layer_k].keys())):
                    if ci > 0:
                        all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
                    else:
                        all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
                
                all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()

                u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
                u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
                u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

                selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)
                
                class_selectivity[layer_k].update({bottleneck_k: selectivity})
    
    
        layer_selectivity = []
        for layer_k, layer_v in class_selectivity.items():
            unit_selectivity = []
            for bottleneck_k, bottleneck_v in class_selectivity[layer_k].items():
                unit_selectivity += class_selectivity[layer_k][bottleneck_k]
            avg_unit_selectivity = sum(unit_selectivity) / len(unit_selectivity)
            layer_selectivity.append(avg_unit_selectivity)
        regularization_term = sum(layer_selectivity) / len(layer_selectivity)

        alpha = args.alpha
        loss = criterion(output, target) - alpha*regularization_term

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.inner_save and i % save_every == 0: 
            save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'train_acc1': acc1,
            'train_acc5': acc5,
            'optimizer' : optimizer.state_dict(),
            }, False, filename='checkpoint_e{}_b{}.pth.tar'.format(epoch+1, i))
        
        if args.save_batch_targets: 
            batch_targets.append(target.cpu())
    
    if args.save_batch_targets: 
        torch.save(batch_targets, EXP_DIR / 'bt_e{}.pt'.format(epoch+1))

    return top1.avg, top5.avg 
        

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg, top5.avg 


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, str(EXP_DIR / filename))
    if is_best:
        shutil.copyfile(str(EXP_DIR / filename), str(EXP_DIR / 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()