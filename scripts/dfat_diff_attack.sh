#!/bin/bash
model=resnet18
dataset=cifar10
gpu=0

python main.py \
--at-method awp_mart \
--method mm_loss \
--model $model \
--dataset $dataset \
--bn 1.0 \
--oh 0.5 \
--cr 0.3 \
--alpha 0.1 \
--adv 0.5 \
--beta 8.0 \
--threshold -2 \
--lr 0.1 \
--epsilon 0.031 \
--num-steps 15 \
--step-size 0.0035 \
--attack-epsilon 0.031 \
--attack-num-steps 20 \
--attack-step-size 0.003 \
--log_tag $model'_'$dataset'__mart_diff_attack' \
--lr_decay_milestones '40,60,80,100' \
--batch_size 512 \
--synthesis_batch_size 512 \
--sample_batch_size 512 \
--gpu $gpu \
--epoch 400 \
