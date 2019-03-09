#!/usr/bin/env bash

## datasets
carla_kitti_dataset_moduletest=../datasets/carla_kitti/carla_kitti_sr_lowquality_moduletest
carla_kitti_dataset_overfit=../datasets/carla_kitti/carla_kitti_sr_lowquality_overfit
carla_kitti_dataset=../datasets/carla_kitti/carla_kitti_sr_lowquality/
sceneflow_dataset=../datasets/sceneflow/
kitti2015_dataset=../datasets/kitti/data_scene_flow/training/
kitti2012_dataset=../datasets/kitti/data_stereo_flow/training/

## pretrained models
pretrained_PSMNet_sceneflow=logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
pretrained_PSMNet_kitti2012=logs/pretrained/PSMNet_pretrained_model_KITTI2012/PSMNet_pretrained_model_KITTI2012.tar
pretrained_PSMNet_kitti2015=logs/pretrained/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
pretrained_EDSR_DIV2K=logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

## GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# prepare: pretrained_Stereo1_Stereo2 (SERVER 95)
# train Stereo1
PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/pretrained_Stereo1_Stereo2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 --loadmodel $pretrained_PSMNet_sceneflow --load_scale 0.5 --half
# train Stereo2
PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/pretrained_Stereo1_Stereo2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 --lossWeights 0.75 0.25 --loadmodel $pretrained_PSMNet_sceneflow --load_scale 1 0.5 --half

## experiment settings
#pretrained_Stereo1=
#pretrained_Stereo2=
#
## experiment 1: SR_SRdisp_compare_carla (SERVER 199)
## test subject: SRdisp > SR
## finetune SRdisp
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SR_SRdisp_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 20 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 16 --batchsize_test $nGPUs --lr 0.0001 10 0.00005 15 0.00002 --loadmodel $pretrained_EDSR_DIV2K --half
## finetune SR
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_SRdisp_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 20 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 16 --batchsize_test $nGPUs  --lr 0.0001 10 0.00005 15 0.00002 --loadmodel $pretrained_EDSR_DIV2K --half

## experiment settings
#pretrained_SR=
#pretrained_SRdisp=
#
## experiment 2: Stereo1_Stereo2_compare_carla (TODO)
## test subject: Stereo2 (PSMNetDown，upbound) > Stereo1 (PSMNet)
## finetune Stereo1
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/Stereo1_Stereo2_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --loadmodel $pretrained_Stereo1 --load_scale 0.5 --half
## finetune Stereo2
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo1_Stereo2_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.75 0.25 --loadmodel $pretrained_Stereo2 --load_scale 1 0.5 --half
#
#
## experiment 3: SRStereo_Stereo1_compare_carla (TODO)
## test subject: SRStereo (baseline) > Stereo2 (PSMNet)
## finetune SRStereo
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_Stereo1_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SR $pretrained_Stereo2 --half
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_Stereo1_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights -1 0.75 0.25 --loadmodel $pretrained_SR $pretrained_Stereo2 --half
#
#
## experiment 4: SRdispStereo_SRStereo_compare_carla (TODO)
## test subject: SRdispStereo (upbound) > SRStereo
## finetune SRdispStereo using same parameters with SRStereo_Stereo1_compare_carla
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereo --dispscale 2 --outputFolder experiments/SRdispStereo_SRStereo_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp $pretrained_Stereo2 --half
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereo --dispscale 2 --outputFolder experiments/SRdispStereo_SRStereo_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights -1 0.75 0.25 --loadmodel $pretrained_SRdisp $pretrained_Stereo2 --half



## experiment 6: SRStereo_PSMNet_compare_kitti (TODO)
## test subject: fintuning SRStereo with KITTI 2015
## parameter settings
#SRStereo_PSMNet_kitti_compare_test_epochs=1300
#SRStereo_PSMNet_kitti_compare_test_Stereo_checkpoint=''
## create baseline PSMNet
#PYTHONPATH=./ python train/Stereo_train.py  --model PSMNet --dispscale 1 --outputFolder experiments/SRStereo_PSMNet_kitti_compare_test --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 256 512 --epochs $SRStereo_PSMNet_kitti_compare_test_epochs --log_every 50 --test_every 10 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 200 0.0001 --loadmodel $pretrained_PSMNet_sceneflow --half
## step 2: fintune SRStereo with KITTI 2015 without updating SR
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_PSMNet_kitti_compare_test --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs $SRStereo_PSMNet_kitti_compare_test_epochs --log_every 50 --test_every 10 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 200 0.0001 --lossWeights -1 0 1 --loadmodel $pretrained_EDSR_DIV2K $SRStereo_PSMNet_kitti_compare_test_Stereo_checkpoint --half


