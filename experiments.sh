#!/usr/bin/env bash

carla_kitti_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/carla_kitti/carla_kitti_sr_lowquality/
sceneflow_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/sceneflow/
kitti2015_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_scene_flow/training/
kitti2012_dataset=/media/omnisky/zcSSD/SR-Stereo/datasets/kitti/data_stereo_flow/training/

## test: SR_SRdisp_compare_test (DONE)
## test subject: Compare SRdisp (EDSR with inputL/R and warpToR/L) with SR (original EDSR_baseline_x2 net)
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SR_SRdisp_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 10 0.00005 15 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_SRdisp_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4  --lr 0.0001 10 0.00005 15 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


## test: SRdisp_cropsize_test (DONE)
## test subject: best crop size for SRdisp
## test method: Fintune SRdisp with carla_kitti dataset
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


# test: Stereo_cropsize_test (DONE)
# test subject: best crop size for Stereo (PSMNet)
# test method: Fintune PSMNet with carla_kitti dataset
# net: Stereo (PSMNet) with same D dimension but different trainCrop
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar


## test: SR_cropsize_test (DONE)
## test subject: best crop size for SR
## test method: Fintune SR with carla_kitti dataset
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


# test: Stereo1_Stereo2_compare_test (TODO)
# test subject: Compare Stereo2 (SR input) with Stereo1 (KITTI-size input PSMNet)
# step 1: training
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --model PSMNet --outputFolder experiments/Stereo1_Stereo2_compare_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --lr 0.001 2 0.0002 4 0.0001 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --model PSMNet --outputFolder experiments/Stereo1_Stereo2_compare_test --dispscale 1 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --lr 0.001 2 0.0002 4 0.0001 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 0.5
# step 2: evaluation
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNetDown --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test 3 --eval_fcn outlier --half --loadmodel logs/experiments/Stereo1_Stereo2_compare_test/Stereo_train/190228024919_PSMNet_loadScale_10_trainCrop_128_1024_batchSize_4_carla_kitti/checkpoint_epoch_0005_it_01000.tar --load_scale 1 0.5
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNet --dispscale 1 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test 3 --eval_fcn outlier --half --loadmodel logs/experiments/Stereo1_Stereo2_compare_test/Stereo_train/190228052018_PSMNet_loadScale_5_trainCrop_128_1024_batchSize_4_carla_kitti/checkpoint_epoch_0005_it_01000.tar --load_scale 0.5
