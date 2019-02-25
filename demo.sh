#!/usr/bin/env bash

carla_kitti_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/carla_kitti/carla_kitti_sr_lowquality/
sceneflow_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/sceneflow/
kitti2015_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_scene_flow/training/
kitti2012_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_stereo_flow/training/

#half-scale PSMNet
#Train PSMNet with sceneflow dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 96 --datapath $sceneflow_dataset --dataset sceneflow --epochs 5 --log_every 50 --test_every 0 --load_scale 0.5 --trainCrop 128 256 --batchsize_train 4 --batchsize_test 4 --lr 0.001 --eval_fcn outlier
#Evaluate with sceneflow dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 96 --datapath $sceneflow_dataset --dataset sceneflow --load_scale 0.5 --batchsize_test 4 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_sceneflow/checkpoint_epoch_10_it_[ITERATION].tar
#Finetune with kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ train/Stereo_train.py --maxdisp 96 --datapath $kitti2015_dataset --dataset kitti2015 --epochs 300 --log_every 50 --test_every 1 --load_scale 0.5 --trainCrop 128 256 --batchsize_train 4 --batchsize_test 4 --lr 0.001 200 0.0001 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_sceneflow/checkpoint_epoch_10_it_[ITERATION].tar
#Evaluate with kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 96 --datapath $kitti2015_dataset --dataset kitti2015 --load_scale 0.5 --batchsize_test 4 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_kitti/checkpoint_epoch_300_it_[ITERATION].tar


#full-scale PSMNet with original cropSize and D dimension trained with carla_kitti
#Fintune PSMNet with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every 0 --batchsize_train 4 --batchsize_test 2 --lr 0.001 --eval_fcn outlier --loadmodel logs/pretrained/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
#Evaluate with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test 2 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_carla_kitti/checkpoint_epoch_10_it_[ITERATION].tar

#half-scale PSMNet baseline trained with carla_kitti
#Fintune PSMNet with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 192 --dispscale 1 --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 0.5 --trainCrop 64 512 --epochs 5 --log_every 50 --test_every 0 --batchsize_train 4 --batchsize_test 4 --lr 0.001 --eval_fcn outlier --loadmodel logs/pretrained/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
#Evaluate with carla_kitti dataset:
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python evaluation/Stereo_eval.py --maxdisp 192 --dispscale 1 --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 0.5 --crop_scale 1 --batchsize_test 4 --eval_fcn outlier --loadmodel logs/Stereo_train/[TRAINING_DATE]_PSMNet_[SUFFIX]_carla_kitti/checkpoint_epoch_10_it_[ITERATION].tar


## test: SR_SRdisp_train_compare_test (DOING...)
## test subject: Compare SRdisp (EDSR with inputL/R and warpToR/L) with SR (original EDSR_baseline_x2 net)
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SRdisp_train.py --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 10 0.00005 15 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4  --lr 0.0001 10 0.00005 15 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


## test: SRdisp_train_cropsize_test (TODO)
## test subject: best crop size for SRdisp
## test method: Fintune SRdisp with carla_kitti dataset
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SRdisp_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SRdisp_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SRdisp_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SRdisp_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SRdisp_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


# test: Stereo_train_cropsize_test (DOING...)
# test subject: best crop size for Stereo (PSMNet)
# test method: Fintune PSMNet with carla_kitti dataset
# net: Stereo (PSMNet) with same D dimension but different trainCrop
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar


## test: SR_train_cropsize_test (DONE)
## test subject: best crop size for SR
## test method: Fintune SR with carla_kitti dataset
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


## test: SR_train_Stereo2_Stereo1_compare_test (DOING...)
## test subject: Compare Stereo2 (SR input) with Stereo1 (KITTI-size input PSMNet)
## step 1: training
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --maxdisp 384 --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 0.5
## step 2: evaluation


