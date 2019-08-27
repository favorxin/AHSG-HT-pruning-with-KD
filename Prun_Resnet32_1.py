#import  pytorch_resnet_cifar10.resnet as resnet
from models import *
from Resnet32_nmodel import CifarResNet
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import time
from torchvision import datasets, transforms
import os

use_cuda=True
best_prec1 = 0.0
batch_size=128


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

#model = torch.nn.DataParallel(resnet.resnet32())
#model=resnet.resnet20(10)
#model=resnet.resnet32(10)
model=resnet.resnet56(10)
#model=resnet.resnet110(10)
#model=resnet.resnet20(100)
#model=resnet.resnet32(100)
#model=resnet.resnet56(100)
#model=resnet.resnet110(100)
#print(model)
model.cuda()

#check = torch.load('D:/Pycharm/Pycharm1/quantization/Prun/68.93_0.7ciafr100_BestResnet32_ASL2BN_epoch1_learning60120160_net.pth.tar')
#print(check['state_dict'])    best   checkpoint
check=torch.load('baseline_newres/b_cifar10_resnet56_0.6_3ci/best.resnet56.pth.tar')

model.load_state_dict(check['state_dict'])
'''for i,(m,n) in enumerate(model.named_parameters()):
    if 'conv' in m:
        print(m)'''


#print(model)
#prune_prob = {'a':[0.9,0.9,0.9]}
#prune_prob = {'a':[0.8,0.8,0.8]}
#prune_prob = {'a':[0.7,0.7,0.7]}
prune_prob = {'a':[0.6,0.6,0.6]}
#prune_prob = {'a':[0.5,0.5,0.5]}
cfg = []                           
cfg_mask = []                      
index_select={}                   


for i,(m,n) in enumerate(model.named_parameters()):

    if 'conv' in m:
        #print(i)
        out_channels = n.data.shape[0]
        #cfg_mask.append(torch.ones(out_channels))
        #cfg.append(out_channels)
        if i <= 32:
            stage = 0
        elif i<= 62:
            stage = 1
        else:
            stage = 2
        keep_prob_stage = prune_prob['a'][stage]
        #print('===>',n.data[13])
        weight_copy=(n.data.abs()).clone().cpu().numpy()
        L2_norm = np.sum(np.sum(weight_copy, axis=1),axis=(1,2))
        #weight_copy = (n.data**2).clone().cpu().numpy()
        #L2_norm = np.sqrt(np.sum(weight_copy, axis=(1, 2, 3)))
        #print(L2_norm)
        prun_num = int(out_channels * (1-keep_prob_stage))
        num_keep=out_channels-prun_num
        arg_max = np.argsort(L2_norm)
        arg_max_rev = arg_max[::-1][:num_keep]
        #print(len(arg_max_rev))
        mask = torch.zeros(out_channels)
        if i==0:
            mask[arg_max_rev.tolist()] = 1
            cfg.append(out_channels)
            for i in range(3):
                cfg_mask.append(mask)
        else:
            mask[arg_max_rev.tolist()] = 1
            for n in range(3):
                cfg_mask.append(mask)
            cfg.append(num_keep)
            if i%2==0:
                index_select[m] = torch.nonzero(torch.Tensor(mask)).view(1, -1).squeeze()

#############new_model=torch.nn.DataParallel(CifarResNet(layers=[5,5,5],index=index_select,rate=cfg))
#new_model=CifarResNet(layers=[3,3,3],index=index_select,rate=cfg,num_classes=10)
#new_model=CifarResNet(layers=[5,5,5],index=index_select,rate=cfg,num_classes=10)
new_model=CifarResNet(layers=[9,9,9],index=index_select,rate=cfg,num_classes=10)
#new_model=CifarResNet(layers=[18,18,18],index=index_select,rate=cfg,num_classes=10)
#new_model=CifarResNet(layers=[3,3,3],index=index_select,rate=cfg,num_classes=100)
#new_model=CifarResNet(layers=[5,5,5],index=index_select,rate=cfg,num_classes=100)
#new_model=CifarResNet(layers=[9,9,9],index=index_select,rate=cfg,num_classes=100)
#new_model=CifarResNet(layers=[18,18,18],index=index_select,rate=cfg,num_classes=100)
#a=torch.randn(4,3,32,32)
#print(new_model)
#print(new_model(a))
for i,[(name0,m0), (name1,m1)] in enumerate(zip(model.named_parameters(), new_model.named_parameters())):
    #print(name0)
    #print(name1)
    if 'conv' in name0:
        if i == 0:
            #print(name0)
            #print(name1)
            m1.data = m0.data.clone()
            continue
        if i % 2 == 1:
            #print(name0)
            #print(name1)
            mask = cfg_mask[i]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.data[idx.tolist(), :, :, :].clone()
            m1.data = w.clone()
            continue
        if i % 2 == 0:
            mask_pre = cfg_mask[i-1]
            mask_later=cfg_mask[i]
            idx_in = np.squeeze(np.argwhere(np.asarray(mask_pre.cpu().numpy())))
            idx_out=np.squeeze(np.argwhere(np.asarray(mask_later.cpu().numpy())))
            if idx_in.size == 1:
                idx_in = np.resize(idx_in, (1,))
                idx_out=np.resize(idx_out,(1,))
            w = m0.data[:, idx_in.tolist(), :, :].clone()
            w=w[idx_out.tolist(),:,:,:]
            m1.data = w.clone()
            continue
    elif 'bn' in name0:
        if 'weight' in name0:
            if i==1:
                m1.data = m0.data.clone()
            else:
                mask = cfg_mask[i]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.data = m0.data[idx.tolist()].clone()
        elif 'bias' in name0:
            if i==2:
                m1.data = m0.data.clone()
            else:
                mask = cfg_mask[i]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.data = m0.data[idx.tolist()].clone()
                #print('====>',m1.data)
    elif 'classifier' in name0 :
        if 'weight' in name0:
            m1.data = m0.data.clone()
        elif 'bias' in name0:
            m1.data = m0.data.clone()

print('===>',new_model)

run={}
m=0
k=0
for j in model.modules():
    if isinstance(j, nn.BatchNorm2d) :
        #print(j)
        run[m]=j
        m+=3
#print(run)
for m1 in new_model.modules():
    if isinstance(m1,nn.BatchNorm2d):
        #print(m1)
        if k==0:
            m1.running_mean = run[k].running_mean.clone()
            m1.running_var = run[k].running_var.clone()
            k+=3
        else:
            mask = cfg_mask[k]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.running_mean = run[k].running_mean[idx.tolist()].clone()
            m1.running_var = run[k].running_var[idx.tolist()].clone()
            k+=3


normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
train_dataset = datasets.CIFAR10(
    root='data/',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataset = datasets.CIFAR10(
    root='data/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

'''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
     pin_memory=True)'''

criterion = nn.CrossEntropyLoss().cuda()

validate(val_loader,new_model,criterion)
#validate(val_loader,model,criterion)

#torch.save(new_model,os.path.join('save',"cifar10_compact_resnet20-0.7_model.pth"))
#torch.save(new_model.state_dict(),os.path.join('save',"cifar10_compact_resnet20-0.7_model_parameters.pth"))