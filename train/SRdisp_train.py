from __future__ import print_function
import torch.utils.data
import os
from models import SR
from evaluation import SRdisp_eval
from utils import myUtils
from train.Train import Train as Base
from models.SR.warp import warp


class Train(Base):
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001,), logEvery=1, testEvery=1, ndisLog=1, Test=None):
        super(Train, self).__init__(trainImgLoader, nEpochs, lr, logEvery, testEvery, ndisLog, Test)

    def _trainIt(self, batch, log):
        super(Train, self)._trainIt(batch, log)
        inputs = batch[4:8] + batch[0:2]
        if log:
            losses, outputs = self.model.train(*inputs, output=True)
            imgs = inputs + outputs

            # save Tensorboard logs to where checkpoint is.
            self.tensorboardLogger.set(self.model.logFolder)
            for imsSide, side in zip((imgs[0::2], imgs[1::2]), ('L', 'R')):
                for name, im in zip(('input', 'dis', 'gt', 'output'), imsSide):
                    self.tensorboardLogger.logFirstNIms('testImages/' + name + side, im, 1,
                                                        global_step=self.global_step, n=self.ndisLog)
        else:
            losses, _ = self.model.train(*inputs, output=False)

        lossesPairs = myUtils.NameValues(('L', 'R'), losses, prefix='loss')

        return lossesPairs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'trainCrop', 'batchsize_test',
         'batchsize_train', 'log_every', 'test_every', 'epochs', 'lr', 'half', 'withMask'],
        description='train or finetune SR net')

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
                                                             loadScale=(args.load_scale[0], args.load_scale[0] / 2),
                                                             mode='training',
                                                             preprocess=False,
                                                             mask=(1, 1, 1, 1))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stage = os.path.join(args.outputFolder, stage) if args.outputFolder is not None else stage
    saveFolderSuffix = myUtils.NameValues(('loadScale', 'trainCrop', 'batchSize'),
                                          (trainImgLoader.loadScale * 10,
                                           trainImgLoader.trainCrop,
                                           args.batchsize_train))
    sr = getattr(SR, 'SRdisp')(args.withMask,
                               cuda=args.cuda, half=args.half, stage=stage,
                               dataset=args.dataset,
                               saveFolderSuffix=saveFolderSuffix.strSuffix())
    if args.loadmodel is not None:
        sr.load(args.loadmodel)

    # Train
    test = SRdisp_eval.Evaluation(testImgLoader=testImgLoader, evalFcn=args.eval_fcn,
                                  ndisLog=args.ndis_log)
    train = Train(trainImgLoader=trainImgLoader, nEpochs=args.epochs, lr=args.lr,
                  logEvery=args.log_every, ndisLog=args.ndis_log,
                  testEvery=args.test_every, Test=test)
    train(model=sr)


if __name__ == '__main__':
    main()
