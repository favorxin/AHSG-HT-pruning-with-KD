#!/bin/bash
#--HTrate 0.7  --k 128 --v 0.1

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90 
resnet20 54
}
 
pruning(){
CUDA_VISIBLE_DEVICES=$1 python pure_distillation.py  data --dataset cifar10 --arch $3 --arch_small $4 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--print_freq 200   --loss $5
}
 

#pruning 1 ./pure_dis/cifar100_vgg11_bn_13_L1_1  L1
# ./vgg_nor/cifar10_vgg19_bn_0.7  ./resnet/cifar10_resnet20_0.7  MSE SmoothL1 

#pruning 1 ./pure_dis/cifar100_vgg11_bn_16  MSE

run13_100(){
(pruning 1 ./pure_dis/cifar100_vgg11_bn_13_L1  L1)&
(pruning 1 ./pure_dis/cifar100_vgg11_bn_13_MSE  MSE)&
(pruning 1 ./pure_dis/cifar100_vgg11_bn_13_SmoothL1  SmoothL1)&
}

#run13_100

#pruning 1 ./pure_dis/cifar100_resnet32_110_MSE resnet110 resnet32 MSE

pruning 1 ./stage_pure/cifar10_resnet32_110_L1 resnet110 resnet32 L1