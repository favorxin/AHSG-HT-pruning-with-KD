import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models
from models import *
import time
from rebuild_vgg import *
from compute_flops import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='vgg11_bn', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--rate', default=0.7, type=float,
                    help='prune rate')
parser.add_argument('--nums', default=1 ,
                    help='save model numbers')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')

args = parser.parse_args()
use_cuda=True
best_prec1 = 0.0
batch_size=128


model = models.__dict__[args.arch]()
#model = vgg13_bn()
#model = vgg16_bn()
#model = vgg19_bn()
if use_cuda:
    model.cuda()

checkpoint = torch.load(args.model)
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
print("(epoch {}) Prec1: {:f}".format(checkpoint['epoch'], best_prec1))

prune_num_0_1 = [58,116,231,461]
prune_num_0_2 = [52,103,205,410]
prune_num_0_3 = [45,90,180,359]
prune_num_0_4 = [39,77,154,308]
prune_num_0_5 = [32,64,128,256]
#vgg11_bn
if args.arch == 'vgg11_bn':
    newmodel = vgg11_bn_rebuild()
    if args.rate == 0.9:
        cfg = [58, 'M', 116, 'M', 231, 231, 'M', 461, 461, 'M', 461, 461, 'M']
    elif args.rate == 0.8:
        cfg = [52, 'M', 103, 'M', 205, 205, 'M', 410, 410, 'M', 410, 410, 'M']
    elif args.rate == 0.7:
        cfg = [45, 'M', 90, 'M', 180, 180, 'M', 359, 359, 'M', 359, 359, 'M']
#vgg13_bn
elif args.arch == 'vgg13_bn':
    newmodel = vgg13_bn_rebuild()
    if args.rate == 0.9:
        cfg = [58, 58, 'M', 116, 116, 'M', 231, 231, 'M', 461, 461, 'M', 461, 461, 'M']
    elif args.rate == 0.8:    
        cfg = [52, 52, 'M', 103, 103, 'M', 205, 205, 'M', 410, 410, 'M', 410, 410, 'M']
    elif args.rate == 0.7:
        cfg = [45, 45, 'M', 90, 90, 'M', 180, 180, 'M', 359, 359, 'M', 359, 359, 'M']
#vgg16_bn
elif args.arch == 'vgg16_bn':
    newmodel = vgg16_bn_rebuild()
    if args.rate == 0.9:
        cfg = [58, 58, 'M', 116, 116, 'M', 231, 231, 231, 'M', 461, 461, 461, 'M', 461, 461, 461]
    elif args.rate == 0.8:
        cfg = [52, 52, 'M', 103, 103, 'M', 205, 205, 205, 'M', 410, 410, 410, 'M', 410, 410, 410]
    elif args.rate == 0.7:
        cfg = [45, 45, 'M', 90, 90, 'M', 180, 180, 180, 'M', 359, 359, 359, 'M', 359, 359, 359]
#vgg19_bn
elif args.arch == 'vgg19_bn':
    newmodel = vgg19_bn_rebuild()
    if args.rate == 0.9:
        cfg = [58, 58, 'M', 116, 116, 'M', 231, 231, 231, 231, 'M', 461, 461, 461, 461, 'M', 461, 461, 461, 461, 'M']
    elif args.rate == 0.8:
        cfg = [52, 52, 'M', 103, 103, 'M', 205, 205, 205, 205, 'M', 410, 410, 410, 410, 'M', 410, 410, 410, 410, 'M']
    elif args.rate == 0.7:
        cfg = [45, 45, 'M', 90, 90, 'M', 180, 180, 180, 180, 'M', 359, 359, 359, 359, 'M', 359, 359, 359, 359, 'M']

start = time.time()

cfg_mask = []
layer_id = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        if out_channels == cfg[layer_id]:
            cfg_mask.append(torch.ones(out_channels))
            layer_id += 1
            continue
        weight_copy = m.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
        arg_max = np.argsort(L1_norm)
        arg_max_rev = arg_max[::-1][:cfg[layer_id]]
        assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
        mask = torch.zeros(out_channels)
        mask[arg_max_rev.tolist()] = 1
        cfg_mask.append(mask)
        layer_id += 1
    elif isinstance(m, nn.MaxPool2d):
        layer_id += 1

#newmodel = vgg(dataset=args.dataset, cfg=cfg)
#newmodel = vgg11_bn_rebuild()
if use_cuda:
    newmodel.cuda()

#i=0
#for name_new, param_new in newmodel.named_parameters():
#    print("i:{} ,name_new: {}, param_new.size: {}".format(i,name_new, param_new.size()))
#    i=i+1

start_mask = torch.ones(3)
layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        if layer_id_in_cfg == len(cfg_mask):
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
            layer_id_in_cfg += 1
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.BatchNorm1d):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()

end = time.time() - start
print('Pruning complete in {:.0f}m {:.0f}s:'.format(end // 60, end % 60))
print("time:",end)

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
    train_data = datasets.CIFAR10('data/', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10('data/', train=False, transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'cifar100':
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
param_nums_old = print_model_param_nums(model)
flops_nums_old = print_model_param_flops(model,32)

test_acc = validate(val_loader,newmodel,criterion)

param_nums_new = print_model_param_nums(newmodel)
flops_nums_new = print_model_param_flops(newmodel,32)

param_rate = round((param_nums_new/param_nums_old),2)
flops_rate = round((flops_nums_new/flops_nums_old),2)

filename = os.path.join(args.save_path, 'checkpoint.{:}_{:}_{:}.pth.tar'.format(args.arch, args.rate, args.nums))

torch.save({'state_dict': newmodel.state_dict(),
            'accuracy': test_acc,
            'parameters_nums': param_nums_new,
            'FLOPs': flops_nums_new,
            }, filename)
#num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save_path, 'prune.{:}_{:}_{:}.pth.txt'.format(args.arch, args.rate, args.nums)), "w") as fp:
    fp.write("Model: \n"+ args.model +"\n")
    fp.write("Number of old model's parameters: \n"+str(param_nums_old)+"M"+"\n")
    fp.write("Number of old model's FLOPs: \n"+str(flops_nums_old)+"G"+"\n")
    fp.write("Number of pruned model's parameters: \n"+str(param_nums_new)+"M"+"\n")
    fp.write("Number of pruned model's FLOPs: \n"+str(flops_nums_new)+"G"+"\n")
    fp.write("Rate of pruned parameters: \n"+str(param_rate)+"\n")
    fp.write("Rate of pruned FLOPs: \n"+str(flops_rate)+"\n")
    fp.write("Test accuracy: \n"+str(test_acc)+"\n")