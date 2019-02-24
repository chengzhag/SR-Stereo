from __future__ import print_function
import torch.utils.data
import os
from models import SR
from tensorboardX import SummaryWriter
from evaluation import SR_eval
from utils import myUtils
from train.Train import Train as Base


class Train(Base):
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001,), logEvery=1, testEvery=1, ndisLog=1, Test=None):
        super(Train, self).__init__(trainImgLoader, nEpochs, lr, logEvery, testEvery, ndisLog, Test)

    def _trainIt(self, batch, log):
        super(Train, self)._trainIt(batch, log)

        batch = batch[0:2] + batch[4:6]

        losses = []
        for input, gt, suffix in zip(batch[2:4], batch[0:2], ('L', 'R')):
            if input is None or gt is None:
                losses.append(None)
                continue
            if log:
                loss, output = self.model.train(input, gt)
                output = myUtils.quantize(output, 1)
                imgs = [input, gt, output]

                # save Tensorboard logs to where checkpoint is.
                writer = SummaryWriter(self.model.logFolder)

                for name, value in [('loss' + suffix, loss), ('lr', self.lrNow)]:
                    writer.add_scalar(self.model.stage + '/trainLosses/' + name, value, self.global_step)

                for name, im in zip(('input', 'gt', 'output'), imgs):
                    myUtils.logFirstNdis(writer, self.model.stage + '/trainImages/' + name + suffix, im, 1,
                                         global_step=self.global_step, n=self.ndisLog)
                writer.close()
            else:
                loss, _ = self.model.train(input, gt)

            losses.append(loss)

        lossesPairs = myUtils.NameValues(('L', 'R'), losses, prefix='loss')
        return lossesPairs


def main():
    parser = myUtils.getBasicParser(
        ['datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'trainCrop', 'batchsize_test',
         'batchsize_train', 'log_every', 'test_every', 'epochs', 'lr'],
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
                                                             loadScale=(args.load_scale, args.load_scale / 2),
                                                             mode='training',
                                                             preprocess=False,
                                                             mask=(1, 1, 0, 0))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    saveFolderSuffix = myUtils.NameValues(('loadScale', 'trainCrop', 'batchSize'),
                                          (trainImgLoader.loadScale * 10,
                                           trainImgLoader.trainCrop,
                                           args.batchsize_train))
    sr = getattr(SR, 'SR')(cuda=args.cuda, stage=stage,
                           dataset=args.dataset,
                           saveFolderSuffix=saveFolderSuffix.strSuffix())
    if args.loadmodel is not None:
        sr.load(args.loadmodel)

    # Train
    test = SR_eval.Evaluation(testImgLoader=testImgLoader, evalFcn=args.eval_fcn,
                              ndisLog=args.ndis_log)
    train = Train(trainImgLoader=trainImgLoader, nEpochs=args.epochs, lr=args.lr,
                  logEvery=args.log_every, ndisLog=args.ndis_log,
                  testEvery=args.test_every, Test=test)
    train(model=sr)


if __name__ == '__main__':
    main()
