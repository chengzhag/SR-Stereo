from __future__ import print_function
import torch.utils.data
import time
import os
from utils import myUtils
import sys
import math


class Train:
    def __init__(self, trainImgLoader, nEpochs, lr=(0.001, ), logEvery=1, testEvery=1, ndisLog=1, Test=None, startEpoch=1, saveEvery=1):
        self.trainImgLoader = trainImgLoader
        self.logEvery = logEvery
        self.testEvery = testEvery
        self.saveEvery = saveEvery
        self.ndisLog = max(ndisLog, 0)
        self.model = None
        self.test = Test
        self.lr = lr
        self.lrNow = lr[0]
        self.nEpochs = nEpochs
        self.startEpoch = startEpoch
        self.global_step = 0
        self.tensorboardLogger = myUtils.TensorboardLogger()
        self.test.tensorboardLogger = self.tensorboardLogger # should be initialized in _trainIt

        
    def _trainIt(self, batch, log):
        return None, None

    def __call__(self, model):
        self.model = model
        # 'model.model is None' means no checkpoint is loaded and presetted maxdisp is used
        if model.model is None:
            model.initModel()

        if self.startEpoch == 1:
            self.model.saveToNew()
        self.log()
        # Train
        ticFull = time.time()

        epoch = None
        batch_idx = None
        lossesAvg = None
        self.global_step = (self.startEpoch - 1) * len(self.trainImgLoader)
        for epoch in range(self.startEpoch, self.nEpochs + 1):
            print('This is %d-th epoch' % (epoch))
            self.lrNow = myUtils.adjustLearningRate(self.model.optimizer, epoch, self.lr)

            # iteration
            totalTrainLoss = 0
            totalAvgIt = 0
            tic = time.time()
            # torch.cuda.empty_cache()
            for batch_idx, batch in enumerate(self.trainImgLoader, 1):
                batch = myUtils.Batch(batch, cuda=self.model.cuda, half=self.model.half)

                self.global_step += 1
                # torch.cuda.empty_cache()

                doLog = self.logEvery > 0 and self.global_step % self.logEvery == 0
                lossesPairs, ims = self._trainIt(batch=batch, log=doLog)
                if ims is not None:
                    for name, im in ims.items():
                        ims[name] = im.cpu()

                if lossesAvg is None:
                    lossesAvg = lossesPairs.copy()
                else:
                    for name in lossesAvg.keys():
                        lossesAvg[name] += lossesPairs[name]

                # save Tensorboard logs to where checkpoint is.
                if doLog:
                    self.tensorboardLogger.set(self.model.logFolder)

                    for name in lossesAvg.keys():
                        lossesAvg[name] /= self.logEvery
                    lossesAvg['lr'] = self.lrNow
                    for name, value in lossesAvg.items():
                        self.tensorboardLogger.writer.add_scalar('trainLosses/' + name, value,
                                                                 self.global_step)
                    lossesAvg = None

                    for name, im in ims.items():
                        if im is not None:
                            self.tensorboardLogger.logFirstNIms('trainImages/' + name, im, 1,
                                                                global_step=self.global_step, n=self.ndisLog)
                if all([not math.isnan(v) for v in lossesPairs.values()]):
                    totalTrainLoss += sum(lossesPairs.values()) / len(lossesPairs.values())
                    totalAvgIt += 1

                timeLeft = (time.time() - tic) / 3600 * (
                        (self.nEpochs - epoch + 1) * len(self.trainImgLoader) - batch_idx)
                printMessage = 'globalIt %d/%d, it %d/%d, epoch %d/%d, %sleft %.2fh' % (
                    self.global_step, len(self.trainImgLoader) * self.nEpochs,
                    batch_idx, len(self.trainImgLoader),
                    epoch, self.nEpochs,
                    lossesPairs.strPrint(''), timeLeft)
                self.tensorboardLogger.writer.add_text('trainPrint/iterations', printMessage, global_step=self.global_step)
                print(printMessage)
                tic = time.time()

            totalTrainLoss /= totalAvgIt
            printMessage = 'epoch %d done, total training loss = %.3f' % (epoch, totalTrainLoss)
            print(printMessage)
            self.tensorboardLogger.writer.add_text('trainPrint/epochs', printMessage, global_step=self.global_step)

            # save
            if (self.saveEvery > 0 and epoch % self.saveEvery == 0)\
                    or (self.saveEvery == 0 and epoch == self.nEpochs):
                model.save(epoch=epoch, iteration=batch_idx,
                           trainLoss=totalTrainLoss)
            # test
            if ((self.testEvery > 0 and epoch % self.testEvery == 0)
                or (self.testEvery == 0 and epoch == self.nEpochs)) \
                    and self.test is not None:
                testScores = self.test(model=self.model, global_step=self.global_step).values()
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
                printMessage = 'Training status: %s' % testReaults.strPrint('')
                self.tensorboardLogger.writer.add_text('trainPrint/testResults', printMessage,
                                                       global_step=self.global_step)
                print(printMessage)
                self.test.log(epoch=epoch, it=batch_idx, global_step=self.global_step,
                              additionalValue=testReaults)
                torch.cuda.empty_cache()

        endMessage = 'Full training time = %.2fh\n' % ((time.time() - ticFull) / 3600)
        print(endMessage)
        self.tensorboardLogger.writer.add_text('trainPrint/timeInfo', endMessage,
                                               global_step=self.global_step)
        self.log(endMessage=endMessage)

    def log(self, additionalValue=(), endMessage=None):
        logFolder = self.model.checkpointFolder
        myUtils.checkDir(logFolder)
        logDir = os.path.join(logFolder, 'train_info.txt')
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
