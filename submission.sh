#!/usr/bin/env bash

## datasets
carla_kitti_dataset_moduletest=../datasets/carla_kitti/carla_kitti_sr_lowquality_moduletest
carla_kitti_dataset_overfit=../datasets/carla_kitti/carla_kitti_sr_lowquality_overfit
carla_kitti_dataset=../datasets/carla_kitti/carla_kitti_sr_lowquality/
sceneflow_dataset=../datasets/sceneflow/
kitti2015_dataset=../datasets/kitti/data_scene_flow/training/
kitti2015_sr_dataset=../datasets/kitti/data_scene_flow_sr/training/
kitti2015_dense_dataset=../datasets/kitti/data_scene_flow_dense/training/
kitti2012_dataset=../datasets/kitti/data_stereo_flow/training/

## dir setting
pretrained_dir=logs/pretrained
experiment_dir=logs/experiments
experiment_bak_dir=logs/experiments_bak

## pretrained models
pretrained_PSMNet_sceneflow=${pretrained_dir}/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
pretrained_PSMNet_kitti2012=${pretrained_dir}/PSMNet_pretrained_model_KITTI2012/PSMNet_pretrained_model_KITTI2012.tar
pretrained_PSMNet_kitti2015=${pretrained_dir}/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
pretrained_EDSR_DIV2K=${pretrained_dir}/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

## GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))


## prepare: pretrain_SR_kitti (SERVER 162)
## finetune SR on kitti2015
PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder submission/pretrain_SR_kitti --datapath $kitti2015_dataset --dataset kitti2015 --trainCrop 64 512 --epochs 6000 --save_every 300 --log_every 50 --test_every 50 --eval_fcn l1 --batchsize_train 64 --lr 0.0001 --loadmodel $pretrained_EDSR_DIV2K --half --subtype subFinal

