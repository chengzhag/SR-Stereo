from __future__ import print_function
import argparse
import torch.utils.data
import time
import os
from models import Stereo
from tensorboardX import SummaryWriter
from evaluation import Stereo_eval
from utils import iteration


class Train:
    def __init__(self, trainImgLoader, logEvery, ndisLog):
        self.trainImgLoader = trainImgLoader
        self.logEvery = logEvery
        self.ndisLog = max(ndisLog, 1)

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
            ndisLog = min(self.ndisLog, disp.size(0))
            disp = disp[:ndisLog, :, :]
            disp[disp > stereo.maxdisp] = stereo.maxdisp
            disp = disp / stereo.maxdisp
            return disp.unsqueeze(1).repeat(1, 3, 1, 1)

        epoch = None
        batch_idx = None
        for epoch in range(1, nEpochs + 1):
            print('This is %d-th epoch' % (epoch))
            adjust_learning_rate(stereo.optimizer, epoch)

            # iteration
            global_step = 1
            tic = time.time()
            for batch_idx, batch in enumerate(self.trainImgLoader, 1):
                batch = [data if data.numel() else None for data in batch]
                if self.logEvery > 0 and global_step % self.logEvery == 0:
                    losses, ouputs = stereo.train(*batch, output=True, kitti=self.trainImgLoader.kitti)
                    lossesPairs = iteration.NameValues('loss', ('L', 'R'), losses)
                    writer.add_scalars('train/losses', lossesPairs.dic(), global_step)
                    if batch[2] is not None:
                        writer.add_images('train/images/gtL', disp2gray(batch[2]), global_step=global_step)
                    if batch[3] is not None:
                        writer.add_images('train/images/gtR', disp2gray(batch[3]), global_step=global_step)
                    if ouputs[0] is not None:
                        writer.add_images('train/images/ouputL', disp2gray(ouputs[0]), global_step=global_step)
                    if ouputs[1] is not None:
                        writer.add_images('train/images/ouputR', disp2gray(ouputs[1]), global_step=global_step)
                else:
                    losses = stereo.train(*batch, output=False, kitti=self.trainImgLoader.kitti)
                    lossesPairs = iteration.NameValues('loss', ('L', 'R'), losses)

                global_step += 1

                try:
                    totalTrainLoss = [(total + batch) if batch is not None else None for total, batch in
                                       zip(totalTrainLoss, losses)]
                except NameError:
                    totalTrainLoss = losses

                timeLeft = (time.time() - tic) / 3600 * ((nEpochs - epoch + 1) * len(self.trainImgLoader) - batch_idx)
                print('it %d/%d, %sleft %.2fh' % (
                    batch_idx, len(self.trainImgLoader) * nEpochs,
                    lossesPairs.str(''), timeLeft))
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
    parser.add_argument('--ndis_log', type=int, default=1,
                        help='number of disparity maps to log')
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        help='evaluation function used in testing')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    trainImgLoader, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset, batchSizes=(12, 11))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, cuda=args.cuda, stage=stage)
    stereo.load(args.loadmodel)

    # Train
    train = Train(trainImgLoader=trainImgLoader, logEvery=args.log_every, ndisLog=args.ndis_log)
    train(stereo=stereo, nEpochs=args.epochs)

    # Test
    test = Stereo_eval.Test(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn, datapath=args.datapath)
    test(stereo=stereo)
    test.log()


if __name__ == '__main__':
    main()
