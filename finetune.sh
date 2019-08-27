#!/bin/bash
#--k 1.5 --print_freq 200 --HTrate 0.9

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90 
resnet20 54
}

pruning(){
CUDA_VISIBLE_DEVICES=$1 python cifar10_resnet_or.py  ./data --dataset cifar10 --arch resnet56 \
--save_path $2  \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.001 --decay 0.0005 --batch_size 64 \
--print_freq 200    --HTrate 0.7  --k 128 --v 0.1   --resume ./baseline_newres/b_cifar10_resnet56_0.6_3ci/best.resnet56.pth.tar
}


pruning 1 ./finetune/cifar10_resnet56_0.6_2