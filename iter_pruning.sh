#!/bin/bash
#--k 1.5 --print_freq 200 --HTrate 0.9

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90 
resnet20 54
}

pruning(){
CUDA_VISIBLE_DEVICES=$1 python cifar10_resnet_or.py  ./data --dataset cifar10 --arch $3 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 64 \
--print_freq 200    --HTrate $4  --k 128 --v 0.1
}

run11(){
(pruning 1 ./vgg_prune/cifar10_vgg11_bn_0.9 vgg11_bn 0.9)&
(pruning 1 ./vgg_prune/cifar10_vgg11_bn_0.8 vgg11_bn 0.8)&
(pruning 1 ./vgg_prune/cifar10_vgg11_bn_0.7 vgg11_bn 0.7)&
}

run11_100(){
(pruning 1 ./vgg_prune_100/cifar100_vgg11_bn_0.9 vgg11_bn 0.9)&
(pruning 1 ./vgg_prune_100/cifar100_vgg11_bn_0.8 vgg11_bn 0.8)&
(pruning 1 ./vgg_prune_100/cifar100_vgg11_bn_0.7 vgg11_bn 0.7)&
}

run13(){
(pruning 1 ./vgg_prune/cifar10_vgg13_bn_0.9 vgg13_bn 0.9)&
(pruning 1 ./vgg_prune/cifar10_vgg13_bn_0.8 vgg13_bn 0.8)&
(pruning 1 ./vgg_prune/cifar10_vgg13_bn_0.7 vgg13_bn 0.7)&
}

run13_100(){
(pruning 1 ./baseline_newres/cifar100_vgg13_bn_0.7 vgg13_bn 0.7)&
(pruning 1 ./baseline_newres/cifar100_vgg11_bn_0.7 vgg11_bn 0.7)&
#(pruning 1 ./vgg_prune_100/cifar10_vgg13_bn_0.7 vgg13_bn 0.7)&
}

run16(){
(pruning 1 ./vgg_prune/cifar10_vgg16_bn_0.9_2 vgg16_bn 0.9)&
(pruning 1 ./vgg_prune/cifar10_vgg16_bn_0.8_2 vgg16_bn 0.8)&
(pruning 1 ./vgg_prune/cifar10_vgg16_bn_0.7_2 vgg16_bn 0.7)&
}


run16_100(){
(pruning 1 ./vgg_prune_100/cifar100_vgg16_bn_0.9 vgg16_bn 0.9)&
(pruning 1 ./vgg_prune_100/cifar100_vgg16_bn_0.8 vgg16_bn 0.8)&
(pruning 1 ./vgg_prune_100/cifar100_vgg16_bn_0.7 vgg16_bn 0.7)&
}

run19(){
(pruning 1 ./vgg_prune/cifar10_vgg19_bn_0.9 vgg19_bn 0.9)&
(pruning 1 ./vgg_prune/cifar10_vgg19_bn_0.8 vgg19_bn 0.8)&
(pruning 1 ./vgg_prune/cifar10_vgg19_bn_0.7 vgg19_bn 0.7)&
}

#run13_100
pruning 1 ./vgg_prune/cifar10_vgg16_bn_0.8_b vgg16_bn 0.8
#pruning 1 ./logs/cifar100_resnet56_0.7 resnet56 0.7

#pruning 1 ./resnet_100/cifar100_resnet110_0.7_a resnet110 0.7
