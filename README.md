# AHSG-HT-pruning-with-KD
Filter pruning with knowledge distillation via hybrid stochastic gradient hard thresholding algorithm

 We implement the hybrid stochastic gradient hard thresholding algorithm in AHSG-HT.py, and it's the optimization algorithm which can be 
found in the paper：Efficient Stochastic Gradient Hard Thresholding. http://papers.nips.cc/paper/7469-efficient-stochastic-gradient-hard-thresholding.

 We modify the code of soft filter pruning:https://github.com/he-y/soft-filter-pruning and implement our method.
 We implement our method in Pytorch0.4.0.

# Requirements
Python3.6
Pytorch 0.4.0

# The pruned models and the others
 We provide some pruned models along with knowledge distillation in /pruning with distillation.
 The else pruned model with knowledge distillation can be found in 链接：https://pan.baidu.com/s/1psja-MthPZ1bLcNU22D7ng 
提取码：w8sx 

# To adopt AHSG-HT optimization
The AHSG algorithm use gradually increasing the mini-batch size from a small number to a large number with step size k. We also have a constant 
momentum strength v=0.1. 
You should modify the dataloader and sampler in the pytorch. You can create a env by using anaconda to install pytorch and replace the 
dataloader.py and sampler.py which can be found in the path of env: 
anaconda/envs/pytorch(this is your env name)/lib/python3.6/site-packages/torch/utils/data/.
You can put our dataloader.py and sampler.py into the above path and use the following code to change the mini-batch size.

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, k=args.k, shuffle=True,
                                                num_workers=args.workers, pin_memory=True)
 args.k is the step size which is added.

from AHSG_HT import AHSG_HT

optimizer = AHSG_HT(net_small.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True, HTrate=state['HTrate'], v=state['v'])

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

 The above code can be found in cifar10_resnet_test.py.

# Knowledge distillation
sh pure_dis.sh

If you want to verify the knowledge distillation, you should run train_original.sh to obtain a well-pretrained large network. Then run pure_dis.sh to guide a small network. And you can found a well-pretrained network in the The pruned models and the others.

# AHSG_HT Pruning
sh iter_pruning.sh

If you want to verify the AHSG_HT pruning algorithm, you can run iter_pruning.sh.

# AHSG_HT Pruning along with Knowledge Distillation
sh distillation.sh

If you want to verify the AHSG_HT pruning with knowledge distillation, you should run distillation.sh to guide the pruned small network's training. And you can found the pruned networks in the The pruned models and the others.

# Introduction
Finally, we use Prun_Resnet.py to reconstruct the ResNet to obtain the pruned model and run vgg_prune.py to to reconstruct the VGG. 

# How to compute the parameters and FLOPs
from compute_flops import all
You can find how to use the code in the vgg_prune.py and Prun_Resnet.py.
