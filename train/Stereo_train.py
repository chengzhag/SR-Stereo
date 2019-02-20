from __future__ import print_function
import argparse
import torch.utils.data
import time
import os
from models import Stereo
from tensorboardX import SummaryWriter
from evaluation import Stereo_eval
from utils import myUtils
import sys

class Train:
    def __init__(self, trainImgLoader, nEpochs, lr=[0.001], logEvery=1, testEvery=1, ndisLog=1, Test=None):
        self.trainImgLoader = trainImgLoader
        self.logEvery = logEvery
        self.testEvery = testEvery
        self.ndisLog = max(ndisLog, 0)
        self.stereo = None
        self.test = Test
        self.lr = lr
        self.nEpochs = nEpochs

    def __call__(self, stereo):
        self.stereo = stereo
        # 'stereo.model is None' means no checkpoint is loaded and presetted maxdisp is used
        if stereo.model is None:
            stereo.initModel()
        self.log()
        
        # Train
        ticFull = time.time()

        epoch = None
        batch_idx = None
        global_step = 0
        for epoch in range(1, self.nEpochs + 1):
            print('This is %d-th epoch' % (epoch))
            lrNow = myUtils.adjustLearningRate(stereo.optimizer, epoch, self.lr)

            # iteration
            totalTrainLoss = 0
            tic = time.time()
            for batch_idx, batch in enumerate(self.trainImgLoader, 1):
                batch = [data if data.numel() else None for data in batch]
                global_step += 1
                torch.cuda.empty_cache()

                if global_step % self.logEvery == 0 and self.logEvery > 0:

                    losses, outputs = stereo.train(*batch, output=True, kitti=self.trainImgLoader.kitti)

                    # save Tensorboard logs to where checkpoint is.
                    lossesPairs = myUtils.NameValues(('L', 'R'), losses, prefix='loss')
                    writer = SummaryWriter(stereo.logFolder)
                    for name, value in lossesPairs.pairs() + [('lr', lrNow), ]:
                        writer.add_scalar(stereo.stage + '/trainLosses/' + name, value, global_step)
                    for name, disp in zip(('gtL', 'gtR', 'ouputL', 'ouputR'), batch[2:4] + outputs):
                        myUtils.logFirstNdis(writer, stereo.stage + '/trainImages/' + name, disp, stereo.maxdisp,
                                             global_step=global_step, n=self.ndisLog)
                    writer.close()
                else:

                    losses = stereo.train(*batch, output=False, kitti=self.trainImgLoader.kitti)

                    lossesPairs = myUtils.NameValues(('L', 'R'), losses, prefix='loss')

                losses = [loss for loss in losses if loss is not None]
                totalTrainLoss += sum(losses) / len(losses)

                timeLeft = (time.time() - tic) / 3600 * ((self.nEpochs - epoch + 1) * len(self.trainImgLoader) - batch_idx)
                print('it %d/%d, %sleft %.2fh' % (
                    global_step, len(self.trainImgLoader) * self.nEpochs,
                    lossesPairs.strPrint(''), timeLeft))
                tic = time.time()

            print('epoch %d done, total training loss = %.3f' % (epoch, totalTrainLoss / len(self.trainImgLoader)))
            # save
            stereo.save(epoch=epoch, iteration=batch_idx,
                        trainLoss=totalTrainLoss / len(self.trainImgLoader))
            # test
            if ((self.testEvery > 0 and epoch % self.testEvery == 0)
                or (self.testEvery == 0 and epoch == self.nEpochs)) \
                    and self.test is not None:
                testScores = self.test(stereo=stereo)
                testScores = [score for score in testScores if score is not None]
                testScore = sum(testScores) / len(testScores)
                try:
                    if testScore <= minTestScore:
                        minTestScore = testScore
                        minTestScoreEpoch = epoch
                except NameError:
                    minTestScore = testScore
                    minTestScoreEpoch = epoch
                testReaults = myUtils.NameValues(
                    ('minTestScore', 'minTestScoreEpoch'), (minTestScore, minTestScoreEpoch))
                print('Training status: %s' % testReaults.strPrint(''))
                self.test.log(epoch=epoch, it=batch_idx, global_step=global_step, additionalValue=testReaults.pairs())

        endMessage = 'Full training time = %.2fh\n' % ((time.time() - ticFull) / 3600)
        print(endMessage)
        self.log(endMessage=endMessage)


    def log(self, additionalValue=(), endMessage=None):
        myUtils.checkDir(self.stereo.saveFolder)
        logDir = os.path.join(self.stereo.saveFolder, 'train_info.txt')
        with open(logDir, "a") as log:
            def writeNotNone(name, value):
                if value is not None: log.write(name + ': ' + str(value) + '\n')

            if endMessage is None:
                log.write('---------------------- %s ----------------------\n\n' % time.asctime(time.localtime(time.time())))

                log.write('python ')
                for arg in sys.argv:
                    log.write(arg + ' ')
                log.write('\n')

                baseInfos = (('data', self.trainImgLoader.datapath),
                             ('load_scale', self.trainImgLoader.loadScale),
                             ('cropScale', self.trainImgLoader.cropScale),
                             ('checkpoint', self.stereo.checkpointDir),
                             ('nEpochs', self.nEpochs),
                             ('lr', self.lr),
                             ('logEvery', self.logEvery),
                             ('testEvery', self.testEvery),
                             ('ndisLog', self.ndisLog),
                             )

                nameValues = baseInfos + additionalValue
                for pairs in (baseInfos, additionalValue):
                    for (name, value) in pairs:
                        writeNotNone(name, value)
                    log.write('\n')

            else:
                log.write(endMessage)
                for pairs in (additionalValue,):
                    for (name, value) in pairs:
                        writeNotNone(name, value)
                    log.write('\n')


def main():
    parser = myUtils.getBasicParser(
        ['maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'crop_scale', 'batchsize_test',
         'batchsize_train', 'log_every', 'test_every', 'epochs', 'lr'],
        description='train or finetune Stereo net')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    trainImgLoader, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                             batchSizes=(args.batchsize_train, args.batchsize_test),
                                                             loadScale=args.load_scale, cropScale=args.crop_scale)

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    saveFolderSuffix = myUtils.NameValues(('loadScale', 'cropScale', 'batchSize'),
                                          (trainImgLoader.loadScale * 10,
                                           trainImgLoader.cropScale * 10,
                                           args.batchsize_train))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, dispScale=args.dispscale, cuda=args.cuda, stage=stage,
                                         dataset=args.dataset,
                                         saveFolderSuffix=saveFolderSuffix.strSuffix())
    if args.loadmodel is not None:
        stereo.load(args.loadmodel)

    # Train
    test = Stereo_eval.Test(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn,
                            ndisLog=args.ndis_log)
    train = Train(trainImgLoader=trainImgLoader, nEpochs=args.epochs, lr=args.lr,
                  logEvery=args.log_every, ndisLog=args.ndis_log,
                  testEvery=args.test_every, Test=test)
    train(stereo=stereo)


if __name__ == '__main__':
    main()
