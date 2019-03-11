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

# experiment settings
pretrained_Stereo2_carla=logs/experiments/pretrain_Stereo1_Stereo2/Stereo_train/190309082616_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_12_lossWeights_0.8_0.2_carla_kitti
pretrained_Stereo1_carla=logs/experiments/pretrain_Stereo1_Stereo2/Stereo_train/190309172438_PSMNet_loadScale_0.5_trainCrop_128_1024_batchSize_12_lossWeights_1_carla_kitti

pretrained_SR_carla=logs/experiments/SR_SRdisp_compare_carla/SR_train/190310140907_SR_loadScale_1_0.5_trainCrop_96_1360_batchSize_4_lossWeights_1_carla_kitti
pretrained_SRdisp_carla=logs/experiments/SR_SRdisp_compare_carla/SR_train/190310141153_SRdisp_loadScale_1_0.5_trainCrop_96_1360_batchSize_4_lossWeights_1_carla_kitti


# prepare: pretrain_Stereo1_Stereo2 (TODO: Stereo2 needs redo)
# train Stereo2
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/pretrain_Stereo1_Stereo2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 4 0.0005 6 0.00025 8 0.000125 --lossWeights 0.75 0.25 --loadmodel $pretrained_PSMNet_sceneflow --load_scale 1 0.5 --half
# train Stereo1
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/pretrain_Stereo1_Stereo2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 4 0.0005 6 0.00025 8 0.000125 --loadmodel $pretrained_PSMNet_sceneflow --load_scale 0.5 --half


## experiment 1: SR_SRdisp_compare_carla (DONE)
## test subject: SRdisp > SR
## finetune SRdisp
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SR_SRdisp_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 4 --batchsize_test $(( 2 * $nGPUs)) --lr 0.0001 10 0.00005 15 0.00002 --loadmodel $pretrained_EDSR_DIV2K
## finetune SR
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_SRdisp_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 4 --batchsize_test $(( 2 * $nGPUs))  --lr 0.0001 10 0.00005 15 0.00002 --loadmodel $pretrained_EDSR_DIV2K


## experiment 2: Stereo1_Stereo2_compare_carla (DONE)
## test subject: Stereo2 (PSMNetDown，upbound) > Stereo1 (PSMNet)
## finetune Stereo1
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/Stereo1_Stereo2_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --loadmodel $pretrained_Stereo1_carla --load_scale 0.5 --half
## finetune Stereo2
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo1_Stereo2_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.75 0.25 --loadmodel $pretrained_Stereo2_carla --load_scale 1 0.5 --half


## experiment 3: SRStereo_Stereo1_compare_carla (SERVER 199)
## test subject: SRStereo (baseline) > Stereo2 (PSMNet)
## finetune SRStereo
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_Stereo1_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights -1 0.75 0.25 --loadmodel $pretrained_SR_carla $pretrained_Stereo2_carla --half
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_Stereo1_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SR_carla $pretrained_Stereo2_carla --half


## experiment 4: SRdispStereo_SRStereo_compare_carla (DONE)
## test subject: SRdispStereo (upbound) > SRStereo
## finetune SRdispStereo using same parameters with SRStereo_Stereo1_compare_carla
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereo --dispscale 2 --outputFolder experiments/SRdispStereo_SRStereo_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights -1 0.75 0.25 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereo --dispscale 2 --outputFolder experiments/SRdispStereo_SRStereo_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half


## experiment 5: SRdispStereoRefine_SRStereo_compare_carla (SERVER 162)
## test subject: SRdispStereoRefine (proposed) > SRStereo
## finetune SRdispStereoRefine using same parameters with SRStereo_Stereo1_compare_carla
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereoRefine --dispscale 2 --outputFolder experiments/SRdispStereoRefine_SRStereo_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -2 --eval_fcn l1 --itRefine 1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 1 0 0 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereoRefine --dispscale 2 --outputFolder experiments/SRdispStereoRefine_SRStereo_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -2 --eval_fcn l1 --itRefine 1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half


## prepare: pretrain_SR_kitti (DONE)
## finetune SR on kitti2015
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/pretrain_SR_kitti --datapath $kitti2015_dataset --dataset kitti2015 --trainCrop 128 1024 --epochs 300 --log_every 50 --test_every 10 --eval_fcn l1 --batchsize_train 4 --batchsize_test $(( 4 * $nGPUs))  --lr 0.0001 --loadmodel $pretrained_EDSR_DIV2K --half

# experiment settings
pretrained_SR_kitti=logs/experiments/pretrain_SR_kitti/SR_train/190310204502_SR_loadScale_1_0.5_trainCrop_128_1024_batchSize_4_lossWeights_1_kitti2015
finetuned_Stereo2_carla=logs/experiments/Stereo1_Stereo2_compare_carla/Stereo_train/190310025752_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_12_lossWeights_0.8_0.2_carla_kitti

## experiment 6: SRStereo_PSMNet_compare_kitti (SERVER 135)
## test subject: fintuning SRStereo with KITTI 2015
## create baseline PSMNet
#PYTHONPATH=./ python train/Stereo_train.py  --model PSMNet --dispscale 1 --outputFolder experiments/SRStereo_PSMNet_compare_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 256 512 --epochs 300 --log_every 50 --test_every 10 --eval_fcn outlier --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 200 0.0001 --loadmodel $pretrained_PSMNet_sceneflow --half
## fintune SRStereo without updating SR
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_PSMNet_compare_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs 300 --log_every 50 --test_every 10 --eval_fcn outlier --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 200 0.0001 --lossWeights -1 0 1 --loadmodel $pretrained_SR_kitti $finetuned_Stereo2_carla --half


## prepare: pretrain_SRdisp_kitti (TODO)
## finetune SRdisp on kitti2015
# TODO: Add script for SRdisp finetuning

## experiment settings
#pretrained_SRdisp_kitti=logs/experiments/pretrain_SRdisp_kitti/SR_train/TODO
#kitti2015_sr_dataset=../datasets/kitti/TODO

## experiment 7: SRdispStereoRefine_PSMNet_compare_kitti (TODO)
## test subject: fintuning SRdispStereoRefine with KITTI 2015
## fintune SRdispStereoRefine with updating SRdisp
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereoRefine --dispscale 2 --outputFolder experiments/SRdispStereoRefine_SRStereo_compare_carla --datapath $kitti2015_sr_dataset --dataset kitti2015_sr --load_scale 1 0.5 --trainCrop 128 1024 --epochs 300 --log_every 50 --test_every 10 --eval_fcn outlier --itRefine 1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half



