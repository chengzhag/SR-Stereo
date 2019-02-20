import argparse
import time
import torch
import os
from models import Stereo
from utils import myUtils
from tensorboardX import SummaryWriter


# Testing for any stereo model including SR-Stereo
class Test:
    def __init__(self, testImgLoader, mode='both', evalFcn='outlier', ndisLog=1):
        self.testImgLoader = testImgLoader
        self.mode = myUtils.assertMode(testImgLoader.kitti, mode)
        self.evalFcn = evalFcn
        self.localtime = None
        self.totalTestScores = None
        self.testTime = None
        self.ndisLog = max(ndisLog, 0)
        self.imgs = []
        self.stereo = None

    def __call__(self, stereo):
        self.stereo = stereo
        scoreUnit = '%' if self.evalFcn == 'outlier' else ''
        tic = time.time()
        ticFull = time.time()

        for batch_idx, batch in enumerate(self.testImgLoader, 1):
            batch = [data if data.numel() else None for data in batch]
            if self.mode == 'right': batch[2] = None

            if batch_idx == 1:
                scores, outputs = stereo.test(*batch, type=self.evalFcn, output=True, kitti=self.testImgLoader.kitti)
                self.imgs = batch[2:4] + outputs
            else:
                scores = stereo.test(*batch, type=self.evalFcn, output=False, kitti=self.testImgLoader.kitti)

            try:
                totalTestScores = [(total + batch) if batch is not None else None for total, batch in
                                   zip(totalTestScores, scores)]
            except NameError:
                totalTestScores = scores
            timeLeft = (time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx)
            scoresPairs = myUtils.NameValues(('L', 'R', 'LTotal', 'RTotal'),
                                             scores + [(score / batch_idx) if score is not None else None
                                                       for score in totalTestScores],
                                             prefix=self.evalFcn)
            print('it %d/%d, %sleft %.2fh' % (
                batch_idx, len(self.testImgLoader),
                scoresPairs.strPrint(scoreUnit), timeLeft))
            tic = time.time()

        totalTestScores = [(score / batch_idx) if score is not None else None
                                          for score in totalTestScores]
        scoresPairs = myUtils.NameValues(('LTotal', 'RTotal'), totalTestScores, prefix=self.evalFcn)

        self.testTime = time.time() - ticFull
        print('Full testing time = %.2fh' % (self.testTime / 3600))
        self.testResults = scoresPairs.pairs()
        self.localtime = time.asctime(time.localtime(time.time()))
        return totalTestScores

    # log file will be saved to where chkpoint file is
    def log(self, epoch=None, it=None, global_step=None, additionalValue=()):
        logDir = os.path.join(self.stereo.checkpointFolder, 'test_results.txt')
        with open(logDir, "a") as log:
            pass
        with open(logDir, "r+") as log:
            def writeNotNone(name, value):
                if value is not None: log.write(name + ': ' + str(value) + '\n')

            log.seek(0)
            logOld = log.read()

            log.seek(0)
            log.write('---------------------- %s ----------------------\n' % self.localtime)
            baseInfos = (('data', self.testImgLoader.datapath ),
                         ('load_scale', self.testImgLoader.loadScale),
                         ('checkpoint', self.stereo.checkpointDir),
                         ('test_type', self.evalFcn),
                         ('test_time', self.testTime),
                         ('epoch', epoch),
                         ('iteration', it),
                         ('global_step', global_step)
                         )
            for pairs in (baseInfos, self.testResults, additionalValue):
                for (name, value) in pairs:
                    writeNotNone(name, value)
                log.write('\n')

            log.write(logOld)

        # save Tensorboard logs to where checkpoint is.
        writer = SummaryWriter(self.stereo.logFolder)
        for name, value in self.testResults:
            writer.add_scalar(self.stereo.stage + '/testLosses/' + name, value, global_step)
        for name, disp in zip(('gtL', 'gtR', 'ouputL', 'ouputR'), self.imgs):
            myUtils.logFirstNdis(writer, self.stereo.stage + '/testImages/' + name, disp, self.stereo.maxdisp,
                                 global_step=global_step, n=self.ndisLog)
        writer.close()


def main():
    parser = myUtils.getBasicParser()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    _, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                batchSizes=(args.batchsize_train, args.batchsize_test),
                                                loadScale=args.load_scale, cropScale=args.crop_scale)

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stereo = getattr(Stereo, args.model)(loadScale=testImgLoader.loadScale, cropScale=testImgLoader.cropScale,
                                         maxdisp=args.maxdisp, cuda=args.cuda, stage=stage)
    stereo.load(args.loadmodel)

    # Test
    test = Test(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn,
                ndisLog=args.ndis_log)
    test(stereo=stereo)
    test.log()


if __name__ == '__main__':
    main()
