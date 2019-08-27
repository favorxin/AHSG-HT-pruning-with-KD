#!/bin/bash
#--HTrate 0.7  --k 128 --v 0.1

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90 
resnet20 54
}
 
pruning(){
CUDA_VISIBLE_DEVICES=$1 python cifar10_resnet_test.py  data --dataset cifar10 --arch $3 --arch_small $4 \
--save_path $2 \
--epochs 250 \
--schedule 1 60 120 160 200 \
--gammas 10 0.2 0.2 0.2 0.5 \
--learning_rate 0.01 --decay 0.0005 --batch_size 64 \
--print_freq 200 --HTrate 0.7  --k 128 --v 0.1
}


#pruning 1 ./resnet/cifar100_resnet20_56_0.7_a resnet56 resnet20
# ./vgg_nor/cifar10_vgg19_bn_0.7  ./resnet/cifar10_resnet20_0.7


run11(){
(pruning 1 ./vgg/cifar10_vgg11_bn_13_0.7_6 vgg13_bn vgg11_bn )&
(pruning 1 ./vgg/cifar10_vgg11_bn_13_0.7_7 vgg13_bn vgg11_bn )&
#(pruning 1 ./vgg/cifar100_vgg11_bn_16_0.7_e vgg16_bn vgg11_bn )&
}

#pruning 1 ./vgg/cifar10_vgg11_bn_13_0.7_6 vgg13_bn vgg11_bn
#run11
pruning 1 ./resnet/cifar10_resnet56_110_0.7_a resnet110 resnet56