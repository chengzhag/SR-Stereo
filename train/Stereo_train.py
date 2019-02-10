from __future__ import print_function
import argparse
import torch.utils.data
import time
from dataloader import listSceneFlowFile
from dataloader import SceneFlowLoader
from models import Stereo
from tensorboardX import SummaryWriter
from evaluation import Stereo_eval


class Train:
    def __init__(self, trainImgLoader, logEvery):
        self.trainImgLoader = trainImgLoader
        self.logEvery = logEvery

    def __call__(self, stereo, nEpochs):
        def adjust_learning_rate(optimizer, epoch):
            lr = 0.001
            print(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # TRAIN
        ticFull = time.time()
        writer = SummaryWriter(stereo.logFolder)
        def disp2gray(disp):
            disp[disp > stereo.maxdisp] = stereo.maxdisp
            disp = disp/stereo.maxdisp
            return disp.unsqueeze(1).repeat(1,3,1,1)

        epoch = None
        batch_idx = None
        for epoch in range(1, nEpochs + 1):
            print('This is %d-th epoch' % (epoch))
            totalTrainLoss = 0
            adjust_learning_rate(stereo.optimizer, epoch)

            # iteration
            global_step = 1
            tic = time.time()
            for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(self.trainImgLoader, 1):
                if stereo.cuda:
                    imgL, imgR, dispL, dispR = imgL.cuda(), imgR.cuda(), dispL.cuda(), dispR.cuda(),
                if self.logEvery > 0 and global_step % self.logEvery == 0:
                    lossAvg, [lossL, lossR], ouputs = stereo.train(imgL, imgR, dispL, dispR, output=True)
                    writer.add_scalars('loss', {'lossAvg': lossAvg, 'lossL': lossL, 'lossR': lossR}, global_step)
                    writer.add_images('dispL', disp2gray(dispL), batch_idx, global_step)
                    writer.add_images('dispR', disp2gray(dispR), batch_idx, global_step)
                    writer.add_images('ouputL', disp2gray(ouputs[0]), batch_idx, global_step)
                    writer.add_images('ouputR', disp2gray(ouputs[1]), batch_idx, global_step)
                else:
                    lossAvg, [lossL, lossR] = stereo.train(imgL, imgR, dispL, dispR, output=False)

                global_step += 1

                totalTrainLoss += lossAvg
                timeLeft = (time.time() - tic) / 3600 * ((nEpochs - epoch + 1) * len(self.trainImgLoader) - batch_idx)
                print('it %d/%d, lossAvg %.2f, lossL %.2f, lossR %.2f, left %.2fh' % (
                    batch_idx, len(self.trainImgLoader) * nEpochs, lossAvg, lossL, lossR, timeLeft))
                tic = time.time()

            print('epoch %d done, total training loss = %.3f' % (epoch, totalTrainLoss / len(self.trainImgLoader)))

            # save
            stereo.save(epoch=epoch, iteration=batch_idx,
                        trainLoss=totalTrainLoss / len(self.trainImgLoader))

        writer.close()
        print('Full training time = %.2fh' % ((time.time() - ticFull) / 3600))


def main():
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
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--both_disparity', type=bool, default=True,
                        help='if train on disparity maps from both views')
    parser.add_argument('--eval_fcn', type=str, default='outlier',
                        help='evaluation function used in testing')
    parser.add_argument('--log_every', type=int, default=10,
                        help='log every log_every iterations')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = listSceneFlowFile.dataloader(
        args.datapath)

    trainImgLoader = torch.utils.data.DataLoader(
        SceneFlowLoader.myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, True),
        batch_size=12, shuffle=True, num_workers=8, drop_last=False)

    testImgLoader = torch.utils.data.DataLoader(
        SceneFlowLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, False),
        batch_size=11, shuffle=False, num_workers=8, drop_last=False)

    # Load model
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, cuda=args.cuda, stage='Stereo_train')
    stereo.load(args.loadmodel)

    # Train
    train = Train(trainImgLoader=trainImgLoader, logEvery=args.log_every)
    train(stereo=stereo, nEpochs=args.epochs)

    # Test
    test = Stereo_eval.Test(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn, datapath=args.datapath)
    test(stereo=stereo)
    test.log()


if __name__ == '__main__':
    main()
