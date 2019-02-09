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
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models.Stereo import Stereo
from tensorboardX import SummaryWriter

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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--bothdisparity', type=bool, default=True,
                    help='if train on disparity maps from both views')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = lt.dataloader(
    args.datapath)

trainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, True),
    batch_size=12, shuffle=True, num_workers=8, drop_last=False)

testImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, False),
    batch_size=11, shuffle=False, num_workers=8, drop_last=False)

stereo = Stereo(maxdisp=args.maxdisp, model=args.model)

if args.cuda:
    stereo.model = nn.DataParallel(stereo.model)
    stereo.model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    stereo.model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in stereo.model.parameters()])))


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
            left = (time.time() - tic) / 3600 * ((args.epochs - epoch + 1) * len(trainImgLoader) - batch_idx - 1)
            print('it %d/%d, lossAvg %.2f, lossL %.2f, lossR %.2f, left %.2fh' % (
                batch_idx, len(trainImgLoader) * args.epochs, lossAvg, lossL, lossR, left))
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
    totalTestLoss = [0, 0, 0]
    tic = time.time()
    for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(testImgLoader):
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
        lossAvg, [lossL, lossR] = stereo.test(imgL, imgR, dispL, dispR, type='l1')
        totalTestLoss = [total + batch for total, batch in zip(totalTestLoss, [lossAvg, lossL, lossR])]
        left = (time.time() - tic) / 3600 * (len(testImgLoader) - batch_idx - 1)
        print(
            'it %d/%d, lossAvg %.2f, lossL %.2f, lossR %.2f, lossTotal %.2f, lossLTotal %.2f, lossRTotal %.2f, left %.2fh' % tuple(
                [batch_idx, len(testImgLoader)] + [lossAvg, lossL, lossR] + [l / (batch_idx + 1) for l in totalTestLoss] + [left]))
        tic = time.time()

    # SAVE test information
    saveDir = args.savemodel + 'testinformation.tar'
    torch.save({
        'test_loss': [l / len(testImgLoader) for l in totalTestLoss],
    }, saveDir)


if __name__ == '__main__':
    main()
