from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listSceneFlowFile
from dataloader import SceneFlowLoader
from models import Stereo
from tensorboardX import SummaryWriter
from evaluation import Stereo_eval

parser = argparse.ArgumentParser(description='Stereo')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='PSMNet',
                    help='select model')
parser.add_argument('--datapath', default='../datasets/sceneflow/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='logs/pretrained/PSMNet_pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default='logs/unamed_PSMNet_sceneflow/',
                    help='save model')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--both_disparity', type=bool, default=True,
                    help='if train on disparity maps from both views')
parser.add_argument('--test_type', type=str, default='outlier',
                    help='evaluation type used in testing')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = listSceneFlowFile.dataloader(
    args.datapath)

trainImgLoader = torch.utils.data.DataLoader(
    SceneFlowLoader.myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, True),
    batch_size=12, shuffle=True, num_workers=8, drop_last=False)

testImgLoader = torch.utils.data.DataLoader(
    SceneFlowLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, False),
    batch_size=11, shuffle=False, num_workers=8, drop_last=False)

stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, cuda=args.cuda)
stereo.load(args.loadmodel)


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    ticFull = time.time()
    if not os.path.exists(args.savemodel):
        os.makedirs(args.savemodel)
    writer = SummaryWriter(os.path.join(args.savemodel, 'logs'))
    # TRAIN
    # TODO: move traing code to a fcn
    for epoch in range(1, args.epochs + 1):
        print('This is %d-th epoch' % (epoch))
        totalTrainLoss = 0
        adjust_learning_rate(stereo.optimizer, epoch)

        # iteration
        tic = time.time()
        for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(trainImgLoader):
            if args.cuda:
                imgL, imgR, dispL, dispR = imgL.cuda(), imgR.cuda(), dispL.cuda(), dispR.cuda(),
            lossAvg, [lossL, lossR] = stereo.train(imgL, imgR, dispL, dispR)
            writer.add_scalars('loss', {'lossAvg': lossAvg, 'lossL': lossL, 'lossR': lossR}, batch_idx)
            totalTrainLoss += lossAvg
            timeLeft = (time.time() - tic) / 3600 * ((args.epochs - epoch + 1) * len(trainImgLoader) - batch_idx - 1)
            print('it %d/%d, lossAvg %.2f, lossL %.2f, lossR %.2f, left %.2fh' % (
                batch_idx, len(trainImgLoader) * args.epochs, lossAvg, lossL, lossR, timeLeft))
            tic = time.time()

        print('epoch %d done, total training loss = %.3f' % (epoch, totalTrainLoss / len(trainImgLoader)))

        # save
        saveDir = os.path.join(args.savemodel, 'checkpoint_%05d.tar' % epoch)
        if not os.path.exists(args.savemodel):
            os.makedirs(args.savemodel)
        torch.save({
            'epoch': epoch,
            'state_dict': stereo.model.state_dict(),
            'train_loss': totalTrainLoss / len(trainImgLoader),
        }, saveDir)

    writer.close()
    print('full training time = %.2f HR' % ((time.time() - ticFull) / 3600))

    # TEST
    totalTestScores, testTime = Stereo_eval.test(stereo=stereo, testImgLoader=testImgLoader, mode='both', type=args.test_type)

    # SAVE test information
    Stereo_eval.logTest(args.datapath, args.savemodel, args.test_type, testTime, (
        ('scoreAvg', totalTestScores[0]),
        ('scoreL', totalTestScores[1]),
        ('scoreR', totalTestScores[2])
    ))


if __name__ == '__main__':
    main()
