import time
import torch
import os
from models import Stereo
from utils import myUtils
from tensorboardX import SummaryWriter
from evaluation.Evaluation import Evaluation as Base

# Evaluation for any stereo model including SR-Stereo
class Evaluation(Base):
    def __init__(self, testImgLoader, mode='both', evalFcn='outlier', ndisLog=1):
        super(Evaluation, self).__init__(testImgLoader, evalFcn='outlier', ndisLog=1)
        self.mode = myUtils.assertMode(testImgLoader.kitti, mode)

    def _evalIt(self, batch, log):
        if self.mode == 'right': batch[2] = None

        if log:
            scores, outputs = self.model.test(*batch, type=self.evalFcn, output=True, kitti=self.testImgLoader.kitti)
            imgs = batch[2:4] + outputs

            # save Tensorboard logs to where checkpoint is.
            writer = SummaryWriter(self.model.logFolder)
            for name, disp in zip(('gtL', 'gtR', 'ouputL', 'ouputR'), imgs):
                myUtils.logFirstNdis(writer, self.model.stage + '/testImages/' + name, disp, self.model.maxdisp,
                                     global_step=1, n=self.ndisLog)
            writer.close()
        else:
            scores = self.model.test(*batch, type=self.evalFcn, output=False, kitti=self.testImgLoader.kitti)

        scoresPairs = myUtils.NameValues(('L', 'R'), scores, prefix=self.evalFcn)
        return scoresPairs


def main():
    parser = myUtils.getBasicParser(['maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'seed', 'eval_fcn',
                                     'ndis_log', 'dataset', 'load_scale', 'batchsize_test'],
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
                                                loadScale=args.load_scale)

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, dispScale=args.dispscale, cuda=args.cuda, stage=stage)
    stereo.load(args.loadmodel)

    # Test
    test = Evaluation(testImgLoader=testImgLoader, mode='both', evalFcn=args.eval_fcn,
                      ndisLog=args.ndis_log)
    test(model=stereo)
    test.log()


if __name__ == '__main__':
    main()
