import time
import torch
import os
from models import SR
from utils import myUtils
from evaluation.Evaluation import Evaluation as Base


# Evaluation for any stereo model including SR-Stereo
class Evaluation(Base):
    def __init__(self, testImgLoader, mode='both', evalFcn='l1', ndisLog=1):
        super(Evaluation, self).__init__(testImgLoader, evalFcn, ndisLog)
        self.mode = myUtils.assertMode(testImgLoader.kitti, mode)

    def _evalIt(self, batch, log):
        batch = batch[0:2] + batch[4:6]
        if self.mode == 'left':
            batch[1] = None
            batch[3] = None
        if self.mode == 'right':
            batch[0] = None
            batch[2] = None

        scores = []
        for input, gt, suffix in zip(batch[2:4], batch[0:2], ('L', 'R')):
            if input is None or gt is None:
                scores.append(None)
                continue
            if log:
                score, output = self.model.test(input, gt, type=self.evalFcn)
                imgs = [input, gt, output]

                # save Tensorboard logs to where checkpoint is.
                self.tensorboardLogger.set(self.model.logFolder)
                for name, im in zip(('input', 'gt', 'output'), imgs):
                    self.tensorboardLogger.logFirstNIms(self.model.stage + '/testImages/' + name + suffix, im, 1,
                                                       global_step=1, n=self.ndisLog)
            else:
                score, _ = self.model.test(input, gt, type=self.evalFcn)

            scores.append(score)

        scoresPairs = myUtils.NameValues(('L', 'R'), scores, prefix=self.evalFcn)
        return scoresPairs


def main():
    parser = myUtils.getBasicParser(
        ['maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
         'ndis_log', 'dataset', 'load_scale', 'batchsize_test', 'half'],
        description='evaluate Stereo net or SR-Stereo net')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    _, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                batchSizes=(0, args.batchsize_test),
                                                loadScale=(args.load_scale, args.load_scale / 2),
                                                mode='testing',
                                                preprocess=False,
                                                mask=(1, 1, 0, 0))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    sr = getattr(SR, 'SR')(cuda=args.cuda, half=args.half, stage=stage,
                           dataset=args.dataset)
    if args.loadmodel is not None:
        sr.load(args.loadmodel)

    # Test
    test = Evaluation(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn,
                      ndisLog=args.ndis_log)
    test(model=sr)
    test.log()


if __name__ == '__main__':
    main()
