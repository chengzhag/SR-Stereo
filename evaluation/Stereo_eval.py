import argparse
import time
import torch
import os
from models import Stereo
from utils import myUtils
from tensorboardX import SummaryWriter


# Testing for any stereo model including SR-Stereo
class Test:
    def __init__(self, testImgLoader, mode='both', evalFcn='outlier', datapath=None, logEvery=1, ndisLog=1):
        self.testImgLoader = testImgLoader
        if self.testImgLoader.kitti:
            self.mode = 'left'
            print(
                'Using dataset KITTI. Evaluation will exclude zero disparity pixels. And only left disparity map will be considered.')
        else:
            self.mode = mode
        self.evalFcn = evalFcn
        self.datapath = datapath
        self.localtime = None
        self.totalTestScores = None
        self.testTime = None
        self.checkpoint = None
        self.logEvery = logEvery
        self.ndisLog = max(ndisLog, 0)

    def __call__(self, stereo):
        self.checkpoint = stereo.checkpoint
        scoreUnit = '%' if self.evalFcn == 'outlier' else ''
        tic = time.time()
        ticFull = time.time()
        # save Tensorboard logs to where checkpoint is.
        chkpointFolder, _ = os.path.split(self.checkpoint)
        logFolder = os.path.join(chkpointFolder, 'logs')
        writer = SummaryWriter(logFolder)

        for batch_idx, batch in enumerate(self.testImgLoader, 1):
            batch = [data if data.numel() else None for data in batch]
            if self.mode == 'right': batch[2] = None

            scores, outputs = stereo.test(*batch, type=self.evalFcn, kitti=self.testImgLoader.kitti)
            outputs = [output.cpu() for output in outputs]
            
            scoresPairs = myUtils.NameValues(self.evalFcn, ('L', 'R'), scores)

            if self.logEvery > 0 and batch_idx % self.logEvery == 0:
                for name, value in scoresPairs.pairs():
                    writer.add_scalar(stereo.stage + '/testLosses/' + name, value, batch_idx)
                for name, disp in zip(('gtL', 'gtR', 'ouputL', 'ouputR'), batch[2:4] + outputs):
                    myUtils.logFirstNdis(writer, stereo.stage + '/testImages/' + name, disp, stereo.maxdisp,
                                         global_step=batch_idx, n=self.ndisLog)

            try:
                totalTestScores = [(total + batch) if batch is not None else None for total, batch in
                                   zip(totalTestScores, scores)]
            except NameError:
                totalTestScores = scores
            timeLeft = (time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx)
            scoresPairs = myUtils.NameValues(self.evalFcn,
                                             ('LTotal', 'RTotal'),
                                             [(score / batch_idx) if score is not None else None for score in
                                                       totalTestScores])
            print('it %d/%d, %sleft %.2fh' % (
                batch_idx, len(self.testImgLoader),
                scoresPairs.str(scoreUnit), timeLeft))
            tic = time.time()

        self.testTime = time.time() - ticFull
        print('Full testing time = %.2fh' % (self.testTime / 3600))
        self.testResults = scoresPairs.pairs()
        self.localtime = time.asctime(time.localtime(time.time()))
        self.totalTestScores = totalTestScores
        return totalTestScores

    # log file will be saved to where chkpoint file is
    def log(self, epoch=None, it=None, additionalValue=()):
        chkpointFolder, _ = os.path.split(self.checkpoint)
        logDir = os.path.join(chkpointFolder, 'test_results.txt')
        with open(logDir, "a") as log:
            pass
        with open(logDir, "r+") as log:
            def writeNotNone(name, value):
                if value is not None: log.write(name + ': ' + str(value) + '\n')

            log.seek(0)
            logOld = log.read()

            log.seek(0)
            log.write('---------------------- %s ----------------------\n' % self.localtime)
            baseInfos = (('data', self.datapath),
                         ('checkpoint', self.checkpoint),
                         ('test_type', self.evalFcn),
                         ('test_time', self.testTime),
                         ('epoch', epoch),
                         ('iteration', it)
                         )
            for pairs in (baseInfos, self.testResults, additionalValue):
                for (name, value) in pairs:
                    writeNotNone(name, value)
                log.write('\n')

            log.write(logOld)


def main():
    parser = argparse.ArgumentParser(description='Stereo')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='PSMNet',
                        help='select model')
    parser.add_argument('--datapath', default='../datasets/sceneflow/',
                        help='datapath')
    parser.add_argument('--loadmodel', default='logs/pretrained/PSMNet_pretrained_sceneflow.tar',
                        help='load model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval_fcn', type=str, default='outlier',
                        help='evaluation function used in testing')
    parser.add_argument('--log_every', type=int, default=1,
                        help='log every log_every iterations')
    parser.add_argument('--ndis_log', type=int, default=1,
                        help='number of disparity maps to log')
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        help='evaluation function used in testing')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    _, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset, batchSizes=(0, 11))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, cuda=args.cuda, stage=stage)
    stereo.load(args.loadmodel)

    # Test
    test = Test(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn, datapath=args.datapath,
                logEvery=args.log_every, ndisLog=args.ndis_log)
    test(stereo=stereo)
    test.log()


if __name__ == '__main__':
    main()
