from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, time_file_str
import models
import numpy as np
#from models import print_log

import random
from AHSG_HT import AHSG_HT

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
#compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1,  help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1,  help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')

parser.add_argument('--HTrate', type=float, default=0.1, help='Hardthreshold rate of model')
parser.add_argument('--k', type=float, default=None, help='k multiply batch_size')
parser.add_argument('--v', type=float, default=0.1, help='data variance')
parser.add_argument('--depth', type=int, default=20, help='data variance')

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

args.prefix = time_file_str()

def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, k=args.k, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, k=args.k, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    #net = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    print_log("=> network :\n {}".format(net), log)

#    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))



    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

#    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
#    optimizer = SGD_HT(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'])
#    optimizer = HSG(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
#    optimizer = AHSG(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, v=state['learning_rate'])
#    optimizer = HSG_HT(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'])    
    optimizer = AHSG_HT(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'], v=state['v'])
    if args.use_cuda:
        net.cuda()
        criterion.cuda()

#    L1_norm_resnet(net.parameters(),args.HTrate)

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
#            recorder = checkpoint['recorder']
#            args.start_epoch = checkpoint['epoch']
#            if args.use_state_dict:
            net.load_state_dict(checkpoint['state_dict'])
#            else:
#                net = checkpoint['state_dict']
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print ('function took %0.3f ms' % ((time2-time1)*1000.0))
        return

    
    filename = os.path.join(args.save_path, 'checkpoint.{:}.pth.tar'.format(args.arch))
    bestname = os.path.join(args.save_path, 'best.{:}.pth.tar'.format(args.arch))
    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    
    best_prec1=0.
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        #current_v = adjust_v(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log)

        # evaluate on validation set
        val_acc_1,   val_los_1   = validate(test_loader, net, criterion, log)
    
        is_best = recorder.update(epoch, train_los, train_acc, val_los_1, val_acc_1)
        #print('val_acc_1',val_acc_1)
        best_prec1 = max(val_acc_1,best_prec1)
        #print('best_prec1',best_prec1)

        '''save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')'''
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename, bestname)
        
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

    log.close()

def L1_norm_resnet(params,rate):
    index_prun = {}

    for idx, p in enumerate(params):
        if idx%3==0 and len(p.size())==4:
            b = []
            prun = int(p.size()[0] * (rate))
            for k in p:
                b.append(torch.norm(k, 1))
            b = torch.FloatTensor(b)
            b = b.cpu().numpy()
            index = b.argsort()[::-1][prun:]
            index_prun[idx]=index
            p.data[index.tolist(), :, :, :] = 0

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    zero1=1
    k_sum=1
    s=0
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

#        if i == (int(50000/args.k)):
#            final=1
        final=1
#        if i == 394:
#            final=1  

        # compute output
        output,_ = model(input_var)
        loss = criterion(output, target_var)
#        print('target.size:',target.size())
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        if k_sum <= args.k:
            optimizer.update_HSG(zero1)
            optimizer.step(None, k_sum,epoch,final)
            k_sum=k_sum+0.1
            zero1=1
            s=s+len(target)
        else:
            optimizer.update_HSG(zero1)
            optimizer.step(None, k_sum,epoch,final)
            s=s+len(target)
        
        
#        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, s, len(train_loader.dataset), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output,_ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

#def save_checkpoint(state, is_best, save_path, filename, bestname):
#    filename = os.path.join(save_path, filename)
#    torch.save(state, filename)
#    if is_best:
#        bestname = os.path.join(save_path, 'model_best.pth.tar')
#        shutil.copyfile(filename, bestname)

def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_v(optimizer, epoch, gammas, schedule):
    """Sets the v to the initial LR decayed by 10 every 30 epochs"""
    v = args.v
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            v = v * gamma
        else:
            break
        for param_group in optimizer.param_groups:
            param_group['v'] = v
    if epoch <= (args.epochs*0.8):
        for param_group in optimizer.param_groups:
            param_group['v'] = v
    else:
        param_group['v'] *= 0
    return v

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


        
if __name__ == '__main__':
    main()
