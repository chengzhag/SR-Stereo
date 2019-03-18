from __future__ import print_function
import torch.utils.data
import os
from models import SR
from evaluation import SR_eval
from utils import myUtils
from train.Train import Train as Base


class Train(Base):
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001,), logEvery=1, testEvery=1, ndisLog=1, Test=None, startEpoch=1, saveEvery=1):
        super(Train, self).__init__(trainImgLoader, nEpochs, lr, logEvery, testEvery, ndisLog, Test, startEpoch, saveEvery)

    def _trainIt(self, batch, log):
        super(Train, self)._trainIt(batch, log)

        losses, outputs = self.model.train(batch.detach(), returnOutputs=log)

        if log:
            imgs = batch.lowResRGBs() + batch.highResRGBs()

            for imsSide, side in zip((imgs[0::2], imgs[1::2]), ('L', 'R')):
                for name, im in zip(('input', 'gt'), imsSide):
                    outputs[name + side] = im.cpu()

        return losses, outputs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'datapath', 'model', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'trainCrop', 'batchsize_test',
         'batchsize_train', 'log_every', 'test_every', 'save_every', 'epochs', 'lr', 'half',
         'withMask', 'randomLR', 'lossWeights', 'resume', 'subtype'],
        description='train or finetune SR net')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    if args.model in ('SR',):
        mask = (1, 1, 0, 0)
    elif args.model in ('SRdisp',):
        mask = (1, 1, 1, 1)
    else:
        raise Exception('Error: No model named \'%s\'!' % args.model)
    trainImgLoader, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                             trainCrop=args.trainCrop,
                                                             batchSizes=(args.batchsize_train, args.batchsize_test),
                                                             loadScale=(args.load_scale[0], args.load_scale[0] / 2),
                                                             mode='training' if args.subtype is None else args.subtype,
                                                             mask=mask,
                                                             randomLR=args.randomLR)

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stage = os.path.join(args.outputFolder, stage) if args.outputFolder is not None else stage
    saveFolderSuffix = myUtils.NameValues(('loadScale', 'trainCrop', 'batchSize','lossWeights'),
                                          (trainImgLoader.loadScale,
                                           trainImgLoader.trainCrop,
                                           args.batchsize_train,
                                           args.lossWeights))
    sr = getattr(SR, args.model)(cuda=args.cuda,
                                 half=args.half, stage=stage,
                                 dataset=args.dataset,
                                 saveFolderSuffix=saveFolderSuffix.strSuffix())
    if hasattr(sr, 'withMask'):
        sr.withMask(args.withMask)
    epoch, iteration = sr.load(args.loadmodel)

    # Train
    test = SR_eval.Evaluation(testImgLoader=testImgLoader, evalFcn=args.eval_fcn,
                              ndisLog=args.ndis_log)
    train = Train(trainImgLoader=trainImgLoader, nEpochs=args.epochs, lr=args.lr,
                  logEvery=args.log_every, ndisLog=args.ndis_log,
                  testEvery=args.test_every, Test=test,
                  startEpoch=epoch + 1 if args.resume else 0, saveEvery=args.save_every)
    train(model=sr)


if __name__ == '__main__':
    main()
