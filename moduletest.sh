#!/usr/bin/env bash

carla_kitti_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/carla_kitti/carla_kitti_sr_lowquality_moduletest/
sceneflow_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/sceneflow/
kitti2015_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_scene_flow/training/
kitti2012_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_stereo_flow/training/
export CUDA_VISIBLE_DEVICES=0,1

# test Stereo_train/eval with PSMNet
echo 'test Stereo_train/eval with PSMNet'
PYTHONPATH=./ python train/Stereo_train.py --outputFolder moduletests --dispscale 0.5 --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 0.25 --trainCrop 128 256 --epochs 3 --log_every 1 --test_every 1  --eval_fcn outlier --batchsize_train 4 --batchsize_test 2 --lr 0.001 --model PSMNet --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar

# test Stereo_train/eval with PSMNet
echo 'test Stereo_train/eval with PSMNetDown'
PYTHONPATH=./ python train/Stereo_train.py --outputFolder moduletests --dispscale 0.5 --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 0.25 0.125 --trainCrop 128 256 --epochs 3 --log_every 1 --test_every 1 --eval_fcn outlier --batchsize_train 4 --batchsize_test 2 --lr 0.001 --model PSMNetDown --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar

# test SR_train/eval with EDSR
echo 'SR_train/eval with EDSR'
PYTHONPATH=./ python train/SR_train.py --outputFolder moduletests --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 0.5 --trainCrop 128 256 --epochs 5 --log_every 1 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 2  --lr 0.0001 --model SR --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

# test SRdisp_train/eval with EDSR
echo 'test SRdisp_train/eval with EDSR'
PYTHONPATH=./ python train/SR_train.py --outputFolder moduletests --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 0.5 --trainCrop 128 256 --epochs 5 --log_every 1 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 2  --lr 0.0001 --model SRdisp --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt




