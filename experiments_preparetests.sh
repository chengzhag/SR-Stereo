#!/usr/bin/env bash

#carla_kitti_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/carla_kitti/carla_kitti_sr_lowquality/
#sceneflow_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/sceneflow/
#kitti2015_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_scene_flow/training/
#kitti2012_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_stereo_flow/training/

#carla_kitti_dataset=/media/omnisky/zc/SR-Stereo/datasets/carla_kitti/carla_kitti_sr_lowquality/
#sceneflow_dataset=/media/omnisky/zc/SR-Stereo/datasets/sceneflow/
#kitti2015_dataset=/media/omnisky/zc/SR-Stereo/datasets/kitti/data_scene_flow/training/
#kitti2012_dataset=/media/omnisky/zc/SR-Stereo/datasets/kitti/data_stereo_flow/training/

carla_kitti_dataset_moduletest=../datasets/carla_kitti/carla_kitti_sr_lowquality_moduletest
carla_kitti_dataset_overfit=../datasets/carla_kitti/carla_kitti_sr_lowquality_overfit
carla_kitti_dataset=../datasets/carla_kitti/carla_kitti_sr_lowquality/
sceneflow_dataset=../datasets/sceneflow/
kitti2015_dataset=../datasets/kitti/data_scene_flow/training/
kitti2012_dataset=../datasets/kitti/data_stereo_flow/training/

export CUDA_VISIBLE_DEVICES=0
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))


# test: SRStereo_lossWeights_test (TODO)
# test subject: weight1 * lossSr + weight2 * lossDispHigh + weight3 * lossDispLow, weight1 + weight2 + weight3 = 1

# parameter settings
SRStereo_lossWeights_test_SR_checkpoint=logs/experiments/SRStereo_lossWeights_test/pretrained/190303022514_SR_loadScale_1_0.5_trainCrop_96_1360_batchSize_4_lossWeights_1_carla_kitti
SRStereo_lossWeights_test_Stereo_checkpoint=logs/experiments/SRStereo_lossWeights_test/pretrained/190304222730_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.8_0.2_carla_kitti

# step 1: module testing
## evaluate pretrained SR with new parameters (PASS)
#PYTHONPATH=./ python evaluation/SR_eval.py --model SR --outputFolder experiments/SR_SRdisp_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --eval_fcn l1 --batchsize_test 4 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint

## pretrain with the same parameters
#PYTHONPATH=./ python train/SR_train.py  --model SR --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.00002 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint

## should have the same behaviour as seperate model
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 1 0 0 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0002 --lossWeights 0 0.75 0.25 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint

# SRStereo_lossWeights_overfit
## try to overfit few images
##PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.001 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0005 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0002 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
##PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.001 --lossWeights 0.2 0.6 0.2 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
##PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0005 --lossWeights 0.2 0.6 0.2 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0002 --lossWeights 0.2 0.6 0.2 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
##PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.001 --lossWeights 0.1 0.675 0.225 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
##PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0005 --lossWeights 0.1 0.675 0.225 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
##PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 500 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0002 --lossWeights 0.1 0.675 0.225 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint

#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 2000 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 1 --batchsize_test 1 --lr 0.0001 1000 0.00005 2000 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 2000 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 1 --batchsize_test 1 --lr 0.0001 1000 0.00005 2000 0.00002 --lossWeights 0.2 0.6 0.2 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 2000 --log_every 5 --test_every -1 --save_every -1 --batchsize_train 1 --batchsize_test 1 --lr 0.0001 1000 0.00005 2000 0.00002 --lossWeights 0.1 0.675 0.225 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#CUDA_VISIBLE_DEVICES=3 PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_overfit --datapath $carla_kitti_dataset_overfit --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 2000 --log_every 50 --test_every -1 --save_every -1 --batchsize_train 1 --batchsize_test 1 --lr 0 --lossWeights 0 0 0 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint

## step 2: training
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.06 0.47 0.47 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.1 0.45 0.45 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.16 0.42 0.42 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0005 --lossWeights 0.06 0.47 0.47 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0005 --lossWeights 0.1 0.45 0.45 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 1 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0005 --lossWeights 0.16 0.42 0.42 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint

#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 3 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 3 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
