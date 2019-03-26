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

    def __call__(self, model, global_step=1):
        self.model = model
        # save Tensorboard logs to where checkpoint is.
        self.tensorboardLogger.set(self.model.logFolder)

        # Evaluation
        tic = time.time()
        ticFull = time.time()
        filter = myUtils.Filter()
        for batch_idx, batch in enumerate(self.testImgLoader, 1):
            batch = myUtils.Batch(batch, cuda=self.model.cuda, half=self.model.half)

            doLog = batch_idx == 1
            scoresPairs, ims = self._evalIt(batch, log=doLog)
            if ims is not None:
                for name, im in ims.items():
                    ims[name] = im.cpu()

            try:
                for name in totalTestScores.keys():
                    totalTestScores[name] += scoresPairs[name]
            except NameError:
                totalTestScores = scoresPairs.copy()

            if doLog:
                for name, im in ims.items():
                    if im is not None:
                        self.tensorboardLogger.logFirstNIms('testImages/' + name, im, 1,
                                                            global_step=global_step, n=self.ndisLog)

            timeLeft = filter((time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx))

            avgTestScores = totalTestScores.copy()
            for name in avgTestScores.keys():
                avgTestScores[name] /= batch_idx

            printMessage = 'it %d/%d, %s%sleft %.2fh' % (
                batch_idx, len(self.testImgLoader),
                scoresPairs.strPrint(), avgTestScores.strPrint(suffix='Avg'), timeLeft)
            print(printMessage)
            self.tensorboardLogger.writer.add_text('testPrint/iterations', printMessage,
                                                   global_step=global_step)

            tic = time.time()

        self.testTime = time.time() - ticFull
        timeInfo = 'Full testing time = %.2fh' % (self.testTime / 3600)
        print(timeInfo)
        self.tensorboardLogger.writer.add_text('testPrint/epochs', timeInfo,
                                               global_step=global_step)
        self.testResults = avgTestScores
        self.localtime = time.asctime(time.localtime(time.time()))
        return avgTestScores

    # log file will be saved to where chkpoint file is
    def log(self, epoch=None, it=None, global_step=None, additionalValue=None):
        logDir = os.path.join(self.model.checkpointFolder, 'test_results.md')

        writeMessage = ''
        writeMessage += '---------------------- %s ----------------------\n\n' % self.localtime
        writeMessage += 'bash param: '
        for arg in sys.argv:
            writeMessage += arg + ' '
        writeMessage += '\n\n'

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
        for pairs, title in zip((baseInfos, self.testResults.items(),
                          additionalValue.items() if additionalValue is not None else ()),
                         ('basic info:', 'test results:', 'additional values:')):
            if len(pairs) > 0:
                writeMessage += title + '\n\n'
                for (name, value) in pairs:
                    if value is not None:
                        writeMessage += '- ' + name + ': ' + str(value) + '\n'
                writeMessage += '\n'

        with open(logDir, "a") as log:
            log.write(writeMessage)

        self.tensorboardLogger.writer.add_text('testPrint/epochs', writeMessage,
                                               global_step=global_step)

        for name, value in self.testResults.items():
            self.tensorboardLogger.writer.add_scalar('testLosses/' + name, value, global_step)
