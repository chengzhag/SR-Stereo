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
submission_dir=logs/submission

## pretrained models
pretrained_PSMNet_sceneflow=${pretrained_dir}/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
pretrained_PSMNet_kitti2012=${pretrained_dir}/PSMNet_pretrained_model_KITTI2012/PSMNet_pretrained_model_KITTI2012.tar
pretrained_PSMNet_kitti2015=${pretrained_dir}/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
pretrained_EDSR_DIV2K=${pretrained_dir}/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

## GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))


pretrained_SR_kitti=${submission_dir}/pretrain_SR_kitti/SR_train/190318163338_SR_loadScale_1_0.5_trainCrop_64_512_batchSize_64_lossWeights_1_kitti2015
pretrained_SRStereo_kitti=${submission_dir}/SRStereo_finetune_kitti/Stereo_train/
finetuned_SRStereo_kitti=${submission_dir}/SRStereo_finetune_kitti/Stereo_train/
finetuned_SRdispStereoRefine_carla=${submission_dir}/SRdispStereoRefine_SRStereo_compare_carla/Stereo_train/190313215524_SRdispStereoRefine_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_12_lossWeights_0.5_0.4_0.1_carla_kitti
pretrained_SRdisp_kitti=${submission_dir}/pretrain_SRdisp_kitti/SR_train/

## prepare: pretrain_SR_kitti (SERVER 162)
## finetune SR on kitti2015
PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder submission/pretrain_SR_kitti --datapath $kitti2015_dataset --dataset kitti2015 --trainCrop 64 512 --epochs 6000 --save_every 300 --log_every 50 --batchsize_train 64 --lr 0.0001 --loadmodel $pretrained_EDSR_DIV2K --half --subtype subFinal

## SRStereo_finetune_kitti (TODO)
## finetune SRStereo initialized with PSMNet pretrained with KITTI and SR finetuned with KITTI without updating SR (TODO)
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder submission/SRStereo_finetune_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs 1200 --save_every 300 --log_every 50 --batchsize_train 12 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --loadmodel $pretrained_SR_kitti $pretrained_PSMNet_kitti2015 --half --subtype subFinal
## finetune SRStereo initialized with prefinetuned SRStereo with updating SR (TODO)
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder submission/SRStereo_finetune_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs 300 --save_every 50 --log_every 50 --batchsize_train 12 --lr 0.0001 --lossWeights 0.5 0 0.5 --loadmodel $pretrained_SRStereo_kitti --half --subtype subFinal
#
#
### prepare: pretrain_SRdisp_kitti (TODO)
#
## generate GTs of SR and dense disparity map with finetuned SRStereo
#PYTHONPATH=./ python submission/SR_sub.py --datapath $kitti2015_dataset --dataset kitti2015 --loadmodel $finetuned_SRStereo_kitti --load_scale 2 1 --subtype subTrainEval --half
#PYTHONPATH=./ python submission/Stereo_sub.py --model SRStereo --dispscale 2 --datapath $kitti2015_dataset --dataset kitti2015 --loadmodel $finetuned_SRStereo_kitti --load_scale 1 --subtype subTrainEval --half
#
## finetune SRdisp on kitti2015_dense: compare different initialization checkpoints (SERVER 135)
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder submission/pretrain_SRdisp_kitti --datapath $kitti2015_dense_dataset --dataset kitti2015_dense --trainCrop 64 2040 --epochs 1500 --save_every 300 --log_every 50 --batchsize_train 16 --lr 0.0005 300 0.0002 500 0.0001 700 0.00005 900 0.00002 1100 0.00001 --loadmodel $finetuned_SRdispStereoRefine_carla --half --subtype subFinal
#
#
## fintune SRdispStereoRefine with updating SRdisp
#PYTHONPATH=./ python train/Stereo_train.py  --model SRdispStereoRefine --dispscale 2 --outputFolder submission/SRdispStereoRefine_finetune_kitti --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 1 --trainCrop 64 512 --epochs 300 --save_every 50 --log_every 50 --itRefine 2 --batchsize_train 12 --lr 0.0001 --lossWeights 0.5 0 0.5 --loadmodel $pretrained_SRdisp_kitti $finetuned_SRStereo_kitti --half --subtype subFinal

