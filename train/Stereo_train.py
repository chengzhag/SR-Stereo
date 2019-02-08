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

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, True),
    batch_size=12, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
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
    start_full_time = time.time()
    if not os.path.exists(args.savemodel):
        os.makedirs(args.savemodel)
    for epoch in range(1, args.epochs + 1):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        adjust_learning_rate(stereo.optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, disp_crop_R) in enumerate(TrainImgLoader):
            start_time = time.time()
            if args.cuda:
                imgL_crop, imgR_crop, disp_crop_L, disp_crop_R = imgL_crop.cuda(), imgR_crop.cuda(), disp_crop_L.cuda(), disp_crop_R.cuda(),
            loss, [lossL, lossR] = stereo.train(imgL_crop, imgR_crop, disp_crop_L, disp_crop_R)
            print('Iter %d training loss = %.3f, lossL = %.3f, lossR = %.3f, time = %.2f' % (batch_idx, loss, lossL, lossR, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))

        # SAVE
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        if not os.path.exists(args.savemodel):
            os.makedirs(args.savemodel)
        torch.save({
            'epoch': epoch,
            'state_dict': stereo.model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(TestImgLoader):
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
        testLoss, [testLossL, testLossR] = stereo.test(imgL, imgR, dispL, dispR, type='l1')
        print('Iter %d test loss = %.3f, left loss = %.3f, right loss = %.3f' % (batch_idx, testLoss, testLossL, testLossR))
        total_test_loss += testLoss

    print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    savefilename = args.savemodel + 'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss / len(TestImgLoader),
    }, savefilename)


if __name__ == '__main__':
    main()
