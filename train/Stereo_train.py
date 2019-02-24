from __future__ import print_function
import torch.utils.data
import time
import os
from models import Stereo
from tensorboardX import SummaryWriter
from evaluation import Stereo_eval
from utils import myUtils
import sys
from train.Train import Train as Base


class Train(Base):
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001, ), logEvery=1, testEvery=1, ndisLog=1, Test=None):
        super(Train, self).__init__(trainImgLoader, nEpochs, lr, logEvery, testEvery, ndisLog, Test)

    def _trainIt(self, batch, log):
        super(Train, self)._trainIt(batch, log)
        if log:
            losses, outputs = self.model.train(*batch, output=True, kitti=self.trainImgLoader.kitti)

            # save Tensorboard logs to where checkpoint is.
            lossesPairs = myUtils.NameValues(('L', 'R'), losses, prefix='loss')
            writer = SummaryWriter(self.model.logFolder)
            for name, value in lossesPairs.pairs() + [('lr', self.lrNow), ]:
                writer.add_scalar(self.model.stage + '/trainLosses/' + name, value, self.global_step)
            for name, disp in zip(('gtL', 'gtR', 'ouputL', 'ouputR'), batch[2:4] + outputs):
                myUtils.logFirstNdis(writer, self.model.stage + '/trainImages/' + name, disp, self.model.maxdisp,
                                     global_step=self.global_step, n=self.ndisLog)
            writer.close()
        else:
            losses = self.model.train(*batch, output=False, kitti=self.trainImgLoader.kitti)

            lossesPairs = myUtils.NameValues(('L', 'R'), losses, prefix='loss')

        return lossesPairs


def main():
    parser = myUtils.getBasicParser(
        ['maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'trainCrop', 'batchsize_test',
         'batchsize_train', 'log_every', 'test_every', 'epochs', 'lr', 'half'],
        description='train or finetune Stereo net')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    trainImgLoader, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                             trainCrop=args.trainCrop,
                                                             batchSizes=(args.batchsize_train, args.batchsize_test),
                                                             loadScale=args.load_scale,
                                                             mode='training')

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    saveFolderSuffix = myUtils.NameValues(('loadScale', 'trainCrop', 'batchSize'),
                                          (trainImgLoader.loadScale * 10,
                                           trainImgLoader.trainCrop,
                                           args.batchsize_train))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, dispScale=args.dispscale,
                                         cuda=args.cuda, half=args.half,
                                         stage=stage,
                                         dataset=args.dataset,
                                         saveFolderSuffix=saveFolderSuffix.strSuffix())
    if args.loadmodel is not None:
        stereo.load(args.loadmodel)

    # Train
    test = Stereo_eval.Evaluation(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn,
                                  ndisLog=args.ndis_log)
    train = Train(trainImgLoader=trainImgLoader, nEpochs=args.epochs, lr=args.lr,
                  logEvery=args.log_every, ndisLog=args.ndis_log,
                  testEvery=args.test_every, Test=test)
    train(model=stereo)


if __name__ == '__main__':
    main()
