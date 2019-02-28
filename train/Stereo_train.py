from __future__ import print_function
import torch.utils.data
import time
import os
from models import Stereo
from evaluation import Stereo_eval
from utils import myUtils
import sys
from train.Train import Train as Base


class Train(Base):
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001,), logEvery=1, testEvery=1, ndisLog=1, Test=None, lossWeights=(1,)):
        super(Train, self).__init__(trainImgLoader, nEpochs, lr, logEvery, testEvery, ndisLog, Test)
        self.lossWeights = lossWeights

    def _trainIt(self, batch, log):
        super(Train, self)._trainIt(batch, log)

        losses, outputs = self.model.train(batch.deattach(),
                                           returnOutputs=log,
                                           kitti=self.trainImgLoader.kitti,
                                           weights=self.lossWeights)
        if log:
            imgs = batch.lowestResDisps()

            for im, side in zip(imgs, ('L', 'R')):
                outputs['gt' + side] = im / self.model.outputMaxDisp

        return losses, outputs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'trainCrop', 'batchsize_test',
         'batchsize_train', 'log_every', 'test_every', 'epochs', 'lr', 'half', 'lossWeights'],
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
    stage = os.path.join(args.outputFolder, stage) if args.outputFolder is not None else stage
    saveFolderSuffix = myUtils.NameValues(('loadScale', 'trainCrop', 'batchSize'),
                                          (trainImgLoader.loadScale[0] * 10,
                                           trainImgLoader.trainCrop,
                                           args.batchsize_train))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, dispScale=args.dispscale,
                                         cuda=args.cuda, half=args.half,
                                         stage=stage,
                                         dataset=args.dataset,
                                         saveFolderSuffix=saveFolderSuffix.strSuffix())
    stereo.load(args.loadmodel)

    # Train
    test = Stereo_eval.Evaluation(testImgLoader=testImgLoader, evalFcn=args.eval_fcn,
                                  ndisLog=args.ndis_log)
    train = Train(trainImgLoader=trainImgLoader, nEpochs=args.epochs, lr=args.lr,
                  logEvery=args.log_every, ndisLog=args.ndis_log,
                  testEvery=args.test_every, Test=test, lossWeights=args.lossWeights)
    train(model=stereo)


if __name__ == '__main__':
    main()
