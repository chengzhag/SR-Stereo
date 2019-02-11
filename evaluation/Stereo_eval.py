import argparse
import time
import torch
import os
from models import Stereo


# Testing for any stereo model including SR-Stereo
class Test:
    def __init__(self, testImgLoader, mode='both', evalFcn='outlier', kitti=False, datapath=None):
        self.testImgLoader = testImgLoader
        self.mode = mode
        self.evalFcn = evalFcn
        self.kitti = kitti
        self.datapath = datapath
        self.localtime = None
        self.totalTestScores = None
        self.testTime = None
        self.checkpoint = None

    def __call__(self, stereo):
        self.checkpoint = stereo.checkpoint
        tic = time.time()

        class NameValues:
            def __init__(self, prefix, suffixes, values):
                self.pairs = []
                for suffix, value in zip(suffixes, values):
                    self.pairs.append((prefix + suffix, value))

            def str(self, unit=''):
                scale = 1
                if unit == '%':
                    scale = 100
                str = ''
                for name, value in self.pairs:
                    str += '%s: %.2f%s, ' % (name, value * scale, unit)
                return str

        scoreUnit = '%' if self.evalFcn == 'outlier' else ''

        if self.mode == 'both':
            totalTestScores = [0, 0, 0]
            tic = time.time()
            for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(self.testImgLoader, 1):
                scoreAvg, [scoreL, scoreR] = stereo.test(imgL, imgR, dispL, dispR, type=self.evalFcn, kitti=self.kitti)
                totalTestScores = [total + batch for total, batch in zip(totalTestScores, (scoreAvg, scoreL, scoreR))]
                timeLeft = (time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx)

                scoresPairs = NameValues(self.evalFcn,
                                         ('Avg', 'L', 'R', 'Total', 'LTotal', 'RTotal'),
                                         [scoreAvg, scoreL, scoreR] + [score / batch_idx for score in totalTestScores])
                print('it %d/%d, %sleft %.2fh' % (
                    batch_idx, len(self.testImgLoader),
                    scoresPairs.str(scoreUnit), timeLeft))
                tic = time.time()

            totalTestScores = [loss / batch_idx for loss in totalTestScores]
            self.testResults = NameValues(self.evalFcn, ('Avg', 'L', 'R'), totalTestScores).pairs

        elif self.mode == 'left' or self.mode == 'right':
            totalTestScore = 0
            tic = time.time()
            for batch_idx, (imgL, imgR, dispGT) in enumerate(self.testImgLoader, 1):
                if self.mode == 'left':
                    score = stereo.test(imgL, imgR, dispL=dispGT, type=self.evalFcn, kitti=self.kitti)
                else:
                    score = stereo.test(imgL, imgR, dispR=dispGT, type=self.evalFcn, kitti=self.kitti)
                totalTestScore += score
                timeLeft = (time.time() - tic) / 3600 * (len(self.testImgLoader) - batch_idx)

                scoresPairs = NameValues(self.evalFcn, ('', 'Total'), (score, totalTestScore / batch_idx))
                print('it %d/%d, %sleft %.2fh' % (
                    batch_idx, len(self.testImgLoader),
                    scoresPairs.str(scoreUnit), timeLeft))
                tic = time.time()

            totalTestScores = totalTestScore / batch_idx
            self.testResults = NameValues(self.evalFcn, (''), (totalTestScores)).pairs

        testTime = time.time() - tic
        print('Full testing time = %.2fh' % (testTime / 3600))
        self.localtime = time.asctime(time.localtime(time.time()))
        self.totalTestScores = totalTestScores
        self.testTime = testTime
        return totalTestScores, testTime

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
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        help='evaluation function used in testing')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # TODO: Test different dataset evaluation
    if args.dataset == 'sceneflow':
        from dataloader import listSceneFlowFiles as listFile
        from dataloader import SceneFlowLoader as fileLoader
    elif args.dataset == 'kitti2012':
        from dataloader import listKitti2012Files as listFile
        from dataloader import KittiLoader as fileLoader
    elif args.dataset == 'kitti2015':
        from dataloader import listKitti2015Files as listFile
        from dataloader import KittiLoader as fileLoader

    _, _, _, _, test_left_img, test_right_img, test_left_disp, test_right_disp = listFile.dataloader(
        args.datapath)

    testImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, False),
        batch_size=11, shuffle=False, num_workers=8, drop_last=False)

    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, cuda=args.cuda)
    stereo.load(args.loadmodel)

    # TEST
    test = Test(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn, datapath=args.datapath)
    test(stereo=stereo)
    test.log()


if __name__ == '__main__':
    main()
