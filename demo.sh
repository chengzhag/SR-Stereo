#!/usr/bin/env bash

#half-scale PSMNet
#Train PSMNet with sceneflow dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 96 --datapath ../datasets/sceneflow/ --dataset sceneflow --epochs 5 --log_every 50 --test_every 0 --load_scale 0.5 --trainCrop 128 256 --batchsize_train 4 --batchsize_test 4 --lr 0.001 --eval_fcn outlier
#Evaluate with sceneflow dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 96 --datapath ../datasets/sceneflow/ --dataset sceneflow --load_scale 0.5 --batchsize_test 4 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_sceneflow/checkpoint_epoch_10_it_[ITERATION].tar
#Finetune with kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ train/Stereo_train.py --maxdisp 96 --datapath ../datasets/kitti/data_scene_flow/training/ --dataset kitti2015 --epochs 300 --log_every 10 --test_every 1 --load_scale 0.5 --trainCrop 128 256 --batchsize_train 4 --batchsize_test 4 --lr 0.001 200 0.0001 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_sceneflow/checkpoint_epoch_10_it_[ITERATION].tar
#Evaluate with kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 96 --datapath ../datasets/kitti/data_scene_flow/training/ --dataset kitti2015 --load_scale 0.5 --batchsize_test 4 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_kitti/checkpoint_epoch_300_it_[ITERATION].tar


#full-scale PSMNet with original cropSize and D dimension trained with carla_kitti
#Fintune PSMNet with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 0 --batchsize_train 4 --batchsize_test 2 --lr 0.001 --eval_fcn outlier --loadmodel logs/pretrained/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
#Evaluate with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti --batchsize_test 2 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_carla_kitti/checkpoint_epoch_10_it_[ITERATION].tar

#half-scale PSMNet baseline trained with carla_kitti
#Fintune PSMNet with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 192 --dispscale 1 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti --load_scale 0.5 --trainCrop 64 512 --epochs 5 --log_every 50 --test_every 0 --batchsize_train 4 --batchsize_test 4 --lr 0.001 --eval_fcn outlier --loadmodel logs/pretrained/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
#Evaluate with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 192 --dispscale 1 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti --load_scale 0.5 --crop_scale 1 --batchsize_test 4 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_carla_kitti/checkpoint_epoch_10_it_[ITERATION].tar


#for testing purpose
#full-scale SRdisp with same settings with EDSR_baseline_x2 but 7 channels input with warped imgs and mask
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SRdisp_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti --trainCrop 96 1360 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4  --lr 0.0002 1 0.0001 3 0.00005 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SRdisp_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti --trainCrop 96 1360 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4  --lr 0.0002 1 0.0001 3 0.00005 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt --withMask
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SR_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti --trainCrop 96 1360 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4  --lr 0.0002 1 0.0001 3 0.00005 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

#for testing purpose
#full-scale PSMNet with same D dimension but different trainCrop trained with carla_kitti
#Fintune PSMNet with carla_kitti dataset:
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 256 512 --epochs 3 --log_every 10 --test_every 1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 1 0.0005 2 0.00001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --half
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 192 688 --epochs 3 --log_every 10 --test_every 1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 1 0.0005 2 0.00001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --half
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 128 1024 --epochs 3 --log_every 10 --test_every 1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 1 0.0005 2 0.00001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --half
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 96 1360 --epochs 3 --log_every 10 --test_every 1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 1 0.0005 2 0.00001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --half
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 80 1632 --epochs 3 --log_every 10 --test_every 1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 1 0.0005 2 0.00001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --half

#for testing purpose
#full-scale EDSR with same settings with EDSR_baseline_x2 but different trainCrop
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SR_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 1 0.00005 3 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SR_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 1 0.00005 3 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SR_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 1 0.00005 3 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SR_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 1 0.00005 3 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/SR_train.py --datapath ../datasets/carla_kitti/carla_kitti_sr_lowquality --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 10 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 1 0.00005 3 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

