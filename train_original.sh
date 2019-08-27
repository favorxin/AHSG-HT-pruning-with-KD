#!/bin/bash
#--HTrate 0.7  --k 128 --v 0.1

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90 
resnet20 54
}
 
pruning(){
CUDA_VISIBLE_DEVICES=$1 python train_or.py  ./data --dataset cifar100 --arch $3 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--print_freq 200 
}


#run16(){
#(pruning 0 ./baseline_newres/cifar100_resnet20  resnet20)&
##(pruning 0 ./baseline_newres/cifar100_resnet32  resnet32 )&
#(pruning 1 ./baseline_newres/cifar100_resnet56 resnet56)&
#(pruning 1 ./baseline_newres/cifar100_resnet110 resnet110)&
#}

pruning 0 ./baseline_newres/cifar100_vgg19  vgg19

#run16
