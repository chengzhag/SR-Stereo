#!/usr/bin/env bash

## datasets
carla_kitti_dataset_moduletest=../datasets/carla_kitti/carla_kitti_sr_lowquality_moduletest
carla_kitti_dataset_overfit=../datasets/carla_kitti/carla_kitti_sr_lowquality_overfit
carla_kitti_dataset=../datasets/carla_kitti/carla_kitti_sr_lowquality/
sceneflow_dataset=../datasets/sceneflow/
kitti2015_dataset=../datasets/kitti/data_scene_flow/training/
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

# experiment settings
pretrained_Stereo2_carla=${experiment_dir}/pretrain_Stereo1_Stereo2/Stereo_train/190312090325_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_12_lossWeights_0.8_0.2_carla_kitti
pretrained_Stereo1_carla=${experiment_dir}/pretrain_Stereo1_Stereo2/Stereo_train/190309172438_PSMNet_loadScale_0.5_trainCrop_128_1024_batchSize_12_lossWeights_1_carla_kitti

pretrained_SR_carla=${experiment_dir}/pretrain_SR_SRdisp_carla/SR_train/190313084045_SR_loadScale_1_0.5_trainCrop_64_2040_batchSize_16_lossWeights_1_carla_kitti
pretrained_SRdisp_carla=${experiment_dir}/pretrain_SR_SRdisp_carla/SR_train/190312223712_SRdisp_loadScale_1_0.5_trainCrop_64_2040_batchSize_16_lossWeights_1_carla_kitti


# prepare: pretrain_Stereo1_Stereo2 (DONE)
# train Stereo2
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/pretrain_Stereo1_Stereo2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 4 0.0005 6 0.00025 8 0.000125 --lossWeights 0.75 0.25 --loadmodel $pretrained_PSMNet_sceneflow --load_scale 1 0.5 --half
# train Stereo1
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/pretrain_Stereo1_Stereo2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 4 0.0005 6 0.00025 8 0.000125 --loadmodel $pretrained_PSMNet_sceneflow --load_scale 0.5 --half


# experiment 1: SR_SRdisp_compare_carla (DONE)
# test subject: SRdisp > SR
# finetune SRdisp
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SR_SRdisp_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 4 --batchsize_test $(( 2 * $nGPUs)) --lr 0.0001 10 0.00005 15 0.00002 --loadmodel $pretrained_EDSR_DIV2K --half
# finetune SR
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_SRdisp_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 2 --eval_fcn l1 --batchsize_train 4 --batchsize_test $(( 2 * $nGPUs))  --lr 0.0001 10 0.00005 15 0.00002 --loadmodel $pretrained_EDSR_DIV2K --half


## prepare: pretrain_SR_SRdisp_carla (DONE)
## note: In experiment 1, SR and SRdisp are trained with full precision. Here we pretrain both with half precision.
## finetune SR
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/pretrain_SR_SRdisp_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 64 2040 --epochs 40 --log_every 50 --test_every 5 --eval_fcn l1 --batchsize_train 16 --batchsize_test $(( 2 * $nGPUs))  --lr 0.0001 25 0.00005 30 0.00002 35 0.00001 --loadmodel $pretrained_EDSR_DIV2K --half
## finetune SRdisp
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/pretrain_SR_SRdisp_carla --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 64 2040 --epochs 40 --log_every 50 --test_every 5 --eval_fcn l1 --batchsize_train 16 --batchsize_test $(( 2 * $nGPUs)) --lr 0.0005 15 0.0002 20 0.0001 25 0.00005 30 0.00002 35 0.00001 --loadmodel $pretrained_EDSR_DIV2K --half


## experiment 2: Stereo1_Stereo2_compare_carla (DONE)
## test subject: Stereo2 (PSMNetDown，upbound) > Stereo1 (PSMNet)
## finetune Stereo1
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/Stereo1_Stereo2_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --loadmodel $pretrained_Stereo1_carla --load_scale 0.5 --half
## finetune Stereo2
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo1_Stereo2_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.75 0.25 --loadmodel $pretrained_Stereo2_carla --load_scale 1 0.5 --half


## experiment 3: SRStereo_Stereo1_compare_carla (DONE)
## test subject: SRStereo (baseline) > Stereo2 (PSMNet)
## finetune SRStereo
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_Stereo1_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 0 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SR_carla $pretrained_Stereo2_carla --half


## experiment 4: SRdispStereo_SRStereo_compare_carla (DONE)
## test subject: SRdispStereo (upbound) > SRStereo
## finetune SRdispStereo using same parameters with SRStereo_Stereo1_compare_carla
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereo --dispscale 2 --outputFolder experiments/SRdispStereo_SRStereo_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 0 --eval_fcn l1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half


## experiment 5: SRdispStereoRefine_SRStereo_compare_carla (DONE)
## test subject: SRdispStereoRefine (proposed) > SRStereo
## finetune SRdispStereoRefine using same parameters with SRStereo_Stereo1_compare_carla
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereoRefine --dispscale 2 --outputFolder experiments/SRdispStereoRefine_SRStereo_compare_carla --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 0 --eval_fcn l1 --itRefine 2 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half


## prepare: pretrain_SR_kitti (DONE)
## finetune SR on kitti2015
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/pretrain_SR_kitti --datapath $kitti2015_dataset --dataset kitti2015 --trainCrop 128 1024 --epochs 300 --log_every 50 --test_every 10 --eval_fcn l1 --batchsize_train 4 --batchsize_test $(( 4 * $nGPUs))  --lr 0.0001 --loadmodel $pretrained_EDSR_DIV2K --half



## kitti experiments

# experiment settings
pretrained_SR_kitti=${experiment_dir}/pretrain_SR_kitti/SR_train/190310204502_SR_loadScale_1_0.5_trainCrop_128_1024_batchSize_4_lossWeights_1_kitti2015
finetuned_Stereo2_carla=${experiment_dir}/Stereo1_Stereo2_compare_carla/Stereo_train/190310025752_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_12_lossWeights_0.8_0.2_carla_kitti

## prepare: pretrain_Stereo2_kitti (TODO)


## experiment 6: SRStereo_PSMNet_compare_kitti (TODO)
## test subject: fintuning SRStereo with KITTI 2015
## create baseline PSMNet (SERVER 135)
#PYTHONPATH=./ python train/Stereo_train.py  --model PSMNet --dispscale 1 --outputFolder experiments/SRStereo_PSMNet_compare_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 256 512 --epochs 300 --log_every 50 --test_every 10 --eval_fcn outlier --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 200 0.0001 --loadmodel $pretrained_PSMNet_sceneflow
## finetune SRStereo without updating SR (SERVER 95)
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_PSMNet_compare_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs 1200 --log_every 50 --test_every 10 --eval_fcn outlier --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 300 0.0005 400 0.0002 500 0.0001 600 0.00005 700 0.00002 800 0.00001 --lossWeights -1 0 1 --loadmodel /media/omnisky/zc/SR-Stereo/SR-Stereo/logs/experiments/SRStereo_PSMNet_compare_kitti/Stereo_train/190314113439_SRStereo_loadScale_1.0_trainCrop_64_512_batchSize_12_lossWeights_-1.0_0.0_1.0_kitti2015 --half --resume
## finetune SRStereo initialized with PSMNet pretrained with KITTI without updating SR (SERVER 199)
PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_PSMNet_compare_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs 1200 --log_every 50 --test_every 10 --eval_fcn outlier --batchsize_train 12 --batchsize_test $nGPUs --lr 0.001 200 0.0005 350 0.0002 500 0.0001 650 0.00005 800 0.00002 950 0.00001 --lossWeights -1 0 1 --loadmodel $pretrained_SR_carla $pretrained_PSMNet_kitti2015 --half
## finetune SRStereo without updating


## prepare: pretrain_SRdisp_kitti (TODO)
## finetune SRdisp on kitti2015
# TODO: Add script for SRdisp finetuning

## experiment settings
#pretrained_SRdisp_kitti=${experiment_dir}/pretrain_SRdisp_kitti/SR_train/TODO
#kitti2015_sr_dataset=../datasets/kitti/TODO

## experiment 7: SRdispStereoRefine_PSMNet_compare_kitti (TODO)
## test subject: fintuning SRdispStereoRefine with KITTI 2015
## fintune SRdispStereoRefine with updating SRdisp
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereoRefine --dispscale 2 --outputFolder experiments/SRdispStereoRefine_SRStereo_compare_carla --datapath $kitti2015_sr_dataset --dataset kitti2015_sr --load_scale 1 0.5 --trainCrop 128 1024 --epochs 300 --log_every 50 --test_every 10 --eval_fcn outlier --itRefine 1 --batchsize_train 12 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $pretrained_SRdisp_carla $pretrained_Stereo2_carla --half



