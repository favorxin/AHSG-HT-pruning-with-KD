import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *
import time
from rebuild_vgg import *

# Prune settings


#model = vgg11_bn()
#model = vgg13_bn()
#model = vgg16_bn()
#model = vgg19_bn()
model = resnet20(100)

model.cuda()

use_cuda=True
best_prec1 = 0.0
batch_size=128
dataset1 = 'cifar100'

checkpoint = torch.load('./pure_dis/cifar100_resnet20_56_MSE/best.resnet56.pth.tar')
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
print("(epoch {}) Prec1: {:f}".format(checkpoint['epoch'], best_prec1))


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=128, shuffle=True, **kwargs)

    model.eval()
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output,_ = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #print(model)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #target = target.cuda(async=True)
        #input_var = torch.autograd.Variable(input, volatile=True).cuda()
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)
        #print(input_var.size())
        #print(target_var.size())


        # compute output
        output,_ = model(input_var)
        #print(output.size())
        #print(output.type())

        loss = criterion(output, target_var.type(torch.LongTensor).cuda())

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data.cpu(), target.cpu())[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30== 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        #print('==>',top1)

    print('*Prec@1 {top1.avg:.3f}' .format(top1=top1))
    return top1.avg

if dataset1 == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
elif dataset1 == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
else:
    assert False, "Unknow dataset : {}".format(dataset1)

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

if dataset1 == 'cifar10':
    train_data = datasets.CIFAR10('data/', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10('data/', train=False, transform=test_transform, download=True)
    num_classes = 10
elif dataset1 == 'cifar100':
    train_data = datasets.CIFAR100('data/', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100('data/', train=False, transform=test_transform, download=True)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                              pin_memory=True)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                             pin_memory=True)


#normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
#train_dataset = datasets.CIFAR10(
#    root='data/',
#    train=True,
#    download=True,
#    transform=transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        normalize,
#    ]))
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#test_dataset = datasets.CIFAR10(
#    root='data/',
#    train=False,
#    download=True,
#    transform=transforms.Compose([
#        transforms.ToTensor(),
#        normalize,
#    ]))
#val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
criterion = nn.CrossEntropyLoss().cuda()

validate(val_loader,model,criterion)
#acc = test(model)