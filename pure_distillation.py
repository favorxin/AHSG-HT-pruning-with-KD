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
#from vgg import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--loss', type=str, choices=['L1', 'MSE', 'SmoothL1'], help='The type of loss function.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--arch_small', metavar='ARCH_small', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')

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

parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
#compress rate
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')

parser.add_argument('--HTrate', type=float, default=0.1, help='Hardthreshold rate of model')
parser.add_argument('--k', type=float, default=None, help='k multiply batch_size')
parser.add_argument('--v', type=float, default=0.1, help='data variance')


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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, k=args.k, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, k=args.k, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    #net = vgg(dataset=args.dataset, depth=19)
    net = models.__dict__[args.arch](num_classes)
    
    net_small = models.__dict__[args.arch_small](num_classes)

    print_log("=>small network :\n {}".format(net_small), log)

    check=torch.load('baseline_newres/cifar10_resnet110/best.resnet110.pth.tar')
    #check=torch.load('finetune/cifar10_resnet56_0.7_f/best.resnet56.pth.tar')
    net.load_state_dict(check['state_dict'])

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.loss == 'L1':
        criterion_s = torch.nn.L1Loss()
    elif args.loss == 'MSE':
        criterion_s = torch.nn.MSELoss()
    elif args.loss == 'SmoothL1':
        criterion_s = torch.nn.SmoothL1Loss()
    #criterion_s = torch.nn.KLDivLoss()

    optimizer = torch.optim.SGD(net_small.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
#    optimizer = SGD_HT(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'])
#    optimizer = HSG(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
#    optimizer = AHSG(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, v=state['learning_rate'])
#    optimizer = HSG_HT(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'])    
#    optimizer = AHSG_HT(net_small.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'], v=state['v'])
    if args.use_cuda:
        net.cuda()
        net_small.cuda()
        criterion.cuda()
        criterion_s.cuda()
        
#    L1_norm_resnet(net_small.parameters(),args.HTrate)

#    i=0
#    for name_new, param_new in net.named_parameters():
#        print("i: {}, name_new: {}, param_new.size: {}".format(i, name_new, param_new.size()))
#        i=i+1

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            if args.use_state_dict:
                net_small.load_state_dict(checkpoint['state_dict'])
            else:
                net_small = checkpoint['state_dict']
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net_small, criterion, log)
        time2 = time.time()
        print ('function took %0.3f ms' % ((time2-time1)*1000.0))
        return

    validate_net(test_loader, net, criterion, log)

    for name_nor, param_nor in net.named_parameters():
        if name_nor == 'classifier2.weight':
            param_w_final = param_nor           
        elif name_nor == 'classifier2.bias':
            param_b_final = param_nor          

    for name_nor, param_nor in net_small.named_parameters():
        if name_nor == 'classifier2.weight':
            param_nor.data = param_w_final 
        elif name_nor == 'classifier2.bias':
            param_nor.data = param_b_final
    
    filename = os.path.join(args.save_path, 'checkpoint.{:}.pth.tar'.format(args.arch))
    bestname = os.path.join(args.save_path, 'best.{:}.pth.tar'.format(args.arch))
    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    
    best_prec1=0.
    for epoch in range(args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        #current_v = adjust_v(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}] [v={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate, args.v) \
                                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, net_small, criterion, criterion_s, optimizer, epoch, log)

        # evaluate on validation set
        val_acc_1,   val_los_1   = validate(test_loader, net_small, criterion, log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los_1, val_acc_1)
        best_prec1 = max(val_acc_1,best_prec1)


        '''save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')'''
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch_small,
            'state_dict': net_small.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename, bestname)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    #torch.save(net_small.state_dict(),os.path.join(args.save_path,"cifar10_resnet110_pretrained_parameters.pth"))


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
def train(train_loader, model, model_small, criterion, criterion_s, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.eval()
    model_small.train()

    zero1=1
    k_sum=1
    s=0
    final=0

#    j=0
#    for name_new, param_new in model.named_parameters():
#        print("j: {}, name_new: {}, param_new.size: {}".format(j, name_new, param_new.size()))
#        j=j+1
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
#        if i == (int(50000/args.k)):
#            final=1
        final=1
#        if i == 394:
#            final=1  


        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
#        output_nor,x_vec=model(input_var)
#        output,x_fc_small = model_small(input_var)
#        x_normal=x_vec.data.clone()
#        #print("input_var.size:",input_var.size())
#        #print("x_normal.size:",x_normal.size())
#        loss_feature = criterion_s(x_fc_small, x_normal)
#        loss = criterion(output,target_var) + loss_feature

        output_nor, x_stage_nor_1, x_stage_nor_2, x_stage_nor_3 = model(input_var)
        output, x_stage_small_1, x_stage_small_2, x_stage_small_3 = model_small(input_var)
        x_stage_nor_1_vec = x_stage_nor_1.data.clone()
        x_stage_nor_2_vec = x_stage_nor_2.data.clone()
        x_stage_nor_3_vec = x_stage_nor_3.data.clone()
#        print("x_stage_nor_1:",x_stage_nor_1)
#        print("x_stage_nor_2:",x_stage_nor_2)
#        print("x_stage_nor_3:",x_stage_nor_3)
#        print("x_stage_small_1:",x_stage_small_1)
#        print("x_stage_small_2:",x_stage_small_2)
#        print("x_stage_small_3:",x_stage_small_3)
        #print("input_var.size:",input_var.size())
        #print("x_normal.size:",x_normal.size())
        loss_feature = criterion_s(x_stage_small_1, x_stage_nor_1_vec) + criterion_s(x_stage_small_2, x_stage_nor_2_vec) +criterion_s(x_stage_small_3, x_stage_nor_3_vec)
        loss = criterion(output,target_var) + loss_feature

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
#        if k_sum <= args.k:
#            optimizer.update_HSG(zero1)
#            optimizer.step(None, k_sum,epoch,final)
#            k_sum=k_sum+0.1
#            zero1=1
#            s=s+len(target)
#        else:
#            optimizer.update_HSG(zero1)
#            optimizer.step(None, k_sum,epoch,final)
#            s=s+len(target)
        
        optimizer.step()

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

def validate(val_loader, model_small, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_small.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output,_,_,_ = model_small(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def validate_net(val_loader, model, criterion, log):
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
        output,_,_,_ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print_log('  **NetTest** Net_Prec@1 {top1.avg:.3f} Net_Prec@5 {top5.avg:.3f} Net_Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


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
