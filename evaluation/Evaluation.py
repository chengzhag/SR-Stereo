import time
import torch
import os
from models import Stereo
from utils import myUtils
from tensorboardX import SummaryWriter

class Evaluation:
    def __init__(self, testImgLoader, evalFcn='outlier', ndisLog=1):
        self.testImgLoader = testImgLoader
        self.evalFcn = evalFcn
        self.localtime = None
        self.totalTestScores = None
        self.testTime = None
        self.ndisLog = max(ndisLog, 0)
        self.model = None

    def _evalIt(self, batch, log):
        pass

    def __call__(self, model):
        self.model = model
        tic = time.time()
        ticFull = time.time()
        scoreUnit = '%' if self.evalFcn == 'outlier' else ''

        for batch_idx, batch in enumerate(self.testImgLoader, 1):
            batch = [data if data.numel() else None for data in batch]

            scoresPairs = self._evalIt(batch, log=(batch_idx == 1))

            try:
                totalTestScores = [(total + batch) for total, batch in zip(totalTestScores, scoresPairs.values())]
            except NameError:
                totalTestScores = scoresPairs.values()

            scoresTotalPairs = myUtils.NameValues(scoresPairs.name(),
                                                  [score / batch_idx for score in totalTestScores],
                                                  suffix='Total')

            timeLeft = (time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx)

            print('it %d/%d, %s%sleft %.2fh' % (
                batch_idx, len(self.testImgLoader),
                scoresPairs.strPrint(scoreUnit), scoresTotalPairs.strPrint(scoreUnit), timeLeft))
            tic = time.time()

        self.testTime = time.time() - ticFull
        print('Full testing time = %.2fh' % (self.testTime / 3600))
        self.testResults = scoresTotalPairs.pairs()
        self.localtime = time.asctime(time.localtime(time.time()))
        return scoresTotalPairs

    # log file will be saved to where chkpoint file is
    def log(self, epoch=None, it=None, global_step=None, additionalValue=()):
        logDir = os.path.join(self.model.checkpointFolder, 'test_results.txt')
        with open(logDir, "a") as log:
            pass
        with open(logDir, "r+") as log:
            def writeNotNone(name, value):
                if value is not None: log.write(name + ': ' + str(value) + '\n')

            log.seek(0)
            logOld = log.read()

            log.seek(0)
            log.write('---------------------- %s ----------------------\n\n' % self.localtime)
            baseInfos = (('data', self.testImgLoader.datapath ),
                         ('loadScale', self.testImgLoader.loadScale),
                         ('trainCrop', self.testImgLoader.trainCrop),
                         ('checkpoint', self.model.checkpointDir),
                         ('evalFcn', self.evalFcn),
                         ('epoch', epoch),
                         ('iteration', it),
                         ('global_step', global_step),
                         ('testTime', self.testTime),
                         )
            for pairs in (baseInfos, self.testResults, additionalValue):
                for (name, value) in pairs:
                    writeNotNone(name, value)
                log.write('\n')

            log.write(logOld)

        # save Tensorboard logs to where checkpoint is.
        writer = SummaryWriter(self.model.logFolder)
        for name, value in self.testResults:
            writer.add_scalar(self.model.stage + '/testLosses/' + name, value, global_step)
        writer.close()