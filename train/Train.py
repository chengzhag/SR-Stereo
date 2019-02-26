from __future__ import print_function
import torch.utils.data
import time
import os
from utils import myUtils
import sys


class Train:
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001, ), logEvery=1, testEvery=1, ndisLog=1, Test=None):
        self.trainImgLoader = trainImgLoader
        self.logEvery = logEvery
        self.testEvery = testEvery
        self.ndisLog = max(ndisLog, 0)
        self.model = None
        self.test = Test
        self.lr = lr
        self.lrNow = lr[0]
        self.nEpochs = nEpochs
        self.global_step = 0
        self.tensorboardLogger = myUtils.TensorboardLogger()
        self.test.tensorboardLogger = self.tensorboardLogger # should be initialized in _trainIt 
        
    def _trainIt(self, batch, log):
        pass

    def __call__(self, model):
        self.model = model
        # 'model.model is None' means no checkpoint is loaded and presetted maxdisp is used
        if model.model is None:
            model.initModel()
        self.log()
        # Train
        ticFull = time.time()

        epoch = None
        batch_idx = None
        self.global_step = 0
        for epoch in range(1, self.nEpochs + 1):
            print('This is %d-th epoch' % (epoch))
            self.lrNow = myUtils.adjustLearningRate(self.model.optimizer, epoch, self.lr)

            # iteration
            totalTrainLoss = 0
            lossesAvg = None
            tic = time.time()
            torch.cuda.empty_cache()
            for batch_idx, batch in enumerate(self.trainImgLoader, 1):
                batch = [(data.half() if self.model.half else data) if data.numel() else None for data in batch]
                batch = [(data.cuda() if self.model.cuda else data) if data is not None else None for data in batch]

                self.global_step += 1
                # torch.cuda.empty_cache()

                doLog = self.global_step % self.logEvery == 0 and self.logEvery > 0
                lossesPairs = self._trainIt(batch=batch, log=doLog)
                if lossesAvg is None:
                    lossesAvg = lossesPairs.values()
                else:
                    lossesAvg = [lossAvg + loss for lossAvg, loss in zip(lossesAvg, lossesPairs.values())]

                # save Tensorboard logs to where checkpoint is.
                if doLog:
                    self.tensorboardLogger.set(self.model.logFolder)
                    lossesAvg = [lossAvg / self.logEvery for lossAvg in lossesAvg]
                    for name, value in zip(lossesPairs.names() + ['lr'], lossesAvg + [self.lrNow]):
                        self.tensorboardLogger.writer.add_scalar('trainLosses/' + name, value,
                                                                 self.global_step)
                    lossesAvg = None

                totalTrainLoss += sum(lossesPairs.values()) / len(lossesPairs.values())

                timeLeft = (time.time() - tic) / 3600 * (
                        (self.nEpochs - epoch + 1) * len(self.trainImgLoader) - batch_idx)
                print('globalIt %d/%d, it %d/%d, epoch %d/%d, %sleft %.2fh' % (
                    self.global_step, len(self.trainImgLoader) * self.nEpochs,
                    batch_idx, len(self.trainImgLoader),
                    epoch, self.nEpochs,
                    lossesPairs.strPrint(''), timeLeft))
                tic = time.time()

            print('epoch %d done, total training loss = %.3f' % (epoch, totalTrainLoss / batch_idx))
            # save
            model.save(epoch=epoch, iteration=batch_idx,
                       trainLoss=totalTrainLoss / len(self.trainImgLoader))
            # test
            if ((self.testEvery > 0 and epoch % self.testEvery == 0)
                or (self.testEvery == 0 and epoch == self.nEpochs)) \
                    and self.test is not None:
                testScores = self.test(model=self.model).values()
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
                self.test.log(epoch=epoch, it=batch_idx, global_step=self.global_step,
                              additionalValue=testReaults.pairs())

        endMessage = 'Full training time = %.2fh\n' % ((time.time() - ticFull) / 3600)
        print(endMessage)
        self.log(endMessage=endMessage)

    def log(self, additionalValue=(), endMessage=None):
        myUtils.checkDir(self.model.saveFolder)
        logDir = os.path.join(self.model.saveFolder, 'train_info.txt')
        with open(logDir, "a") as log:
            def writeNotNone(name, value):
                if value is not None: log.write(name + ': ' + str(value) + '\n')

            if endMessage is None:
                log.write(
                    '---------------------- %s ----------------------\n\n' % time.asctime(time.localtime(time.time())))

                log.write('python ')
                for arg in sys.argv:
                    log.write(arg + ' ')
                log.write('\n\n')

                baseInfos = (('data', self.trainImgLoader.datapath),
                             ('load_scale', self.trainImgLoader.loadScale),
                             ('trainCrop', self.trainImgLoader.trainCrop),
                             ('checkpoint', self.model.checkpointDir),
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
