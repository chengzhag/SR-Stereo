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

export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))


## test: SR_SRdisp_compare_test (DONE)
## test subject: Compare SRdisp (EDSR with inputL/R and warpToR/L) with SR (original EDSR_baseline_x2 net)
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SR_SRdisp_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 10 0.00005 15 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_SRdisp_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 96 1360 --epochs 20 --log_every 50 --test_every 1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4  --lr 0.0001 10 0.00005 15 0.00002 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


## test: SRdisp_cropsize_test (DONE)
## test subject: best crop size for SRdisp
## test method: Fintune SRdisp with carla_kitti dataset
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SRdisp --outputFolder experiments/SRdisp_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


## test: Stereo_cropsize_test (DONE)
## test subject: best crop size for Stereo (PSMNet)
## test method: Fintune PSMNet with carla_kitti dataset
## net: Stereo (PSMNet) with same D dimension but different trainCrop
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/Stereo_cropsize_test --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 4 0.0001 --eval_fcn 6_outlier --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar


## test: SR_cropsize_test (DONE)
## test subject: best crop size for SR
## test method: Fintune SR with carla_kitti dataset
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 256 512 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 192 688 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 96 1360 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt
#PYTHONPATH=./ python train/SR_train.py --model SR --outputFolder experiments/SR_cropsize_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 80 1632 --epochs 5 --log_every 50 --test_every -1 --eval_fcn l1 --batchsize_train 4 --batchsize_test 4 --lr 0.0001 --loadmodel logs/pretrained/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt


# test: Stereo2_lossWeights_test (DONE)
# test subject: lossDisp = weight1 * lossDispHigh + weight2 * lossDispLow, weight1 + weight2 = 1
## step 1: training
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo2_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 5 0.0001 --lossWeights 1 0 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo2_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 5 0.0001 --lossWeights 0 1 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo2_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 5 0.0001 --lossWeights 0.5 0.5 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo2_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 5 0.0001 --lossWeights 0.25 0.75 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo2_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --trainCrop 128 1024 --epochs 5 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test 2 --lr 0.001 2 0.0002 5 0.0001 --lossWeights 0.75 0.25 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1 0.5
## step 2: evaluation
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNetDown --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test $nGPUs --eval_fcn outlier --loadmodel logs/experiments/Stereo2_lossWeights_test/Stereo_train/190304143122_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.0_1.0_carla_kitti --load_scale 1 0.5 --half --resume
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNetDown --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test $nGPUs --eval_fcn outlier --loadmodel logs/experiments/Stereo2_lossWeights_test/Stereo_train/190304170526_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.5_0.5_carla_kitti --load_scale 1 0.5 --half --resume
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNetDown --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test $nGPUs --eval_fcn outlier --loadmodel logs/experiments/Stereo2_lossWeights_test/Stereo_train/190304193920_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.2_0.8_carla_kitti --load_scale 1 0.5 --half --resume
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNetDown --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test $nGPUs --eval_fcn outlier --loadmodel logs/experiments/Stereo2_lossWeights_test/Stereo_train/190304222730_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.8_0.2_carla_kitti --load_scale 1 0.5 --half --resume

# test: Stereo1_Stereo2_compare_test (TODO)
# test subject: Compare Stereo2 (SR input) with Stereo1 (KITTI-size input PSMNet)
## step 1: training
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/Stereo1_Stereo2_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every 0 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.001 2 0.0002 6 0.0001 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo1_Stereo2_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 10 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.001 2 0.0002 6 0.0001 --lossWeights 0.75 0.25 --loadmodel logs/pretrained/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar --load_scale 1 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --dispscale 1 --outputFolder experiments/Stereo1_Stereo2_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 2 --log_every 50 --test_every 1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0002 --loadmodel logs/experiments/Stereo1_Stereo2_compare_test/Stereo_train/190305102004_PSMNet_loadScale_0.5_trainCrop_128_1024_batchSize_4_lossWeights_1_carla_kitti --load_scale 0.5 --half
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --dispscale 2 --outputFolder experiments/Stereo1_Stereo2_compare_test --datapath $carla_kitti_dataset --dataset carla_kitti  --trainCrop 128 1024 --epochs 2 --log_every 50 --test_every 1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0002 --lossWeights 0.75 0.25 --loadmodel logs/experiments/Stereo1_Stereo2_compare_test/Stereo_train/190305153219_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.8_0.2_carla_kitti --load_scale 1 0.5 --half
## step 2: evaluation
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNetDown --dispscale 2 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test $nGPUs --eval_fcn outlier --loadmodel logs/experiments/Stereo1_Stereo2_compare_test/Stereo_train/190305153219_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.8_0.2_carla_kitti --load_scale 1 0.5 --half --resume
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model PSMNet --dispscale 1 --datapath $carla_kitti_dataset --dataset carla_kitti --batchsize_test $nGPUs --eval_fcn outlier --loadmodel logs/experiments/Stereo1_Stereo2_compare_test/Stereo_train/190303115921_PSMNet_loadScale_0.5_trainCrop_128_1024_batchSize_4_lossWeights_1_carla_kitti --load_scale 0.5 --half --resume


# test: SRStereo_lossWeights_test (TODO)
# test subject: weight1 * lossSr + weight2 * lossDispHigh + weight3 * lossDispLow, weight1 + weight2 + weight3 = 1
# parameter settings
#SRStereo_lossWeights_test_SR_checkpoint=logs/experiments/SRStereo_lossWeights_test/pretrained/190303022514_SR_loadScale_1_0.5_trainCrop_96_1360_batchSize_4_lossWeights_1_carla_kitti
#SRStereo_lossWeights_test_Stereo_checkpoint=logs/experiments/SRStereo_lossWeights_test/pretrained/190304222730_PSMNetDown_loadScale_1.0_0.5_trainCrop_128_1024_batchSize_4_lossWeights_0.8_0.2_carla_kitti
## step 2: training
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 3 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.0001 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint
#PYTHONPATH=./ python train/Stereo_train.py  --model SRStereo --dispscale 2 --outputFolder experiments/SRStereo_lossWeights_test --datapath $carla_kitti_dataset --dataset carla_kitti --load_scale 1 0.5 --trainCrop 128 1024 --epochs 3 --log_every 50 --test_every -1 --batchsize_train 4 --batchsize_test $nGPUs --lr 0.00002 --lossWeights 0.5 0.375 0.125 --loadmodel $SRStereo_lossWeights_test_SR_checkpoint $SRStereo_lossWeights_test_Stereo_checkpoint

