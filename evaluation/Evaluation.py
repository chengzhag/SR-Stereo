import time
import os
from utils import myUtils
import sys

class Evaluation:
    def __init__(self, testImgLoader, evalFcn='outlier', ndisLog=1):
        self.testImgLoader = testImgLoader
        self.evalFcn = evalFcn
        self.localtime = None
        self.totalTestScores = None
        self.testTime = None
        self.ndisLog = max(ndisLog, 0)
        self.model = None
        self.tensorboardLogger = myUtils.TensorboardLogger() # should be initialized in _evalIt

    def _evalIt(self, batch, log):
        return None, None

    def __call__(self, model):
        self.model = model
        tic = time.time()
        ticFull = time.time()
        scoreUnit = '%' if 'outlier' in self.evalFcn else ''

        for batch_idx, batch in enumerate(self.testImgLoader, 1):
            batch = myUtils.Batch(batch, cuda=self.model.cuda, half=self.model.half)

            doLog = batch_idx == 1
            scoresPairs, ims = self._evalIt(batch, log=doLog)
            if ims is not None:
                ims = [im.cpu() for im in ims]

            try:
                for name in totalTestScores.keys():
                    totalTestScores[name] += scoresPairs[name]
            except NameError:
                totalTestScores = scoresPairs.copy()

            # save Tensorboard logs to where checkpoint is.
            if doLog:
                self.tensorboardLogger.set(self.model.logFolder)
                for name, im in ims.items():
                    if im is not None:
                        self.tensorboardLogger.logFirstNIms('testImages/' + name, im, 1,
                                                            global_step=1, n=self.ndisLog)

            timeLeft = (time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx)

            avgTestScores = totalTestScores.copy()
            for name in avgTestScores.keys():
                avgTestScores[name] /= batch_idx

            print('it %d/%d, %s%sleft %.2fh' % (
                batch_idx, len(self.testImgLoader),
                scoresPairs.strPrint(scoreUnit), avgTestScores.strPrint(scoreUnit, suffix='Avg'), timeLeft))
            tic = time.time()

        self.testTime = time.time() - ticFull
        print('Full testing time = %.2fh' % (self.testTime / 3600))
        self.testResults = avgTestScores
        self.localtime = time.asctime(time.localtime(time.time()))
        return avgTestScores

    # log file will be saved to where chkpoint file is
    def log(self, epoch=None, it=None, global_step=None, additionalValue=None):
        logDir = os.path.join(self.model.checkpointFolder, 'test_results.txt')
        with open(logDir, "a") as log:
            def writeNotNone(name, value):
                if value is not None: log.write(name + ': ' + str(value) + '\n')

            log.write('---------------------- %s ----------------------\n\n' % self.localtime)

            log.write('python ')
            for arg in sys.argv:
                log.write(arg + ' ')
            log.write('\n\n')

            baseInfos = (('data', self.testImgLoader.datapath),
                         ('loadScale', self.testImgLoader.loadScale),
                         ('trainCrop', self.testImgLoader.trainCrop),
                         ('checkpoint', self.model.checkpointDir),
                         ('evalFcn', self.evalFcn),
                         ('epoch', epoch),
                         ('iteration', it),
                         ('global_step', global_step),
                         ('testTime', self.testTime),
                         )
            for pairs in (baseInfos, self.testResults.items(),
                          additionalValue.items() if additionalValue is not None else ()):
                for (name, value) in pairs:
                    writeNotNone(name, value)
                log.write('\n')

        # save Tensorboard logs to where checkpoint is.
        for name, value in self.testResults.items():
            self.tensorboardLogger.writer.add_scalar('testLosses/' + name, value, global_step)
