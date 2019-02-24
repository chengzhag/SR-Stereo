from utils import myUtils
import argparse
import time
import torch
import os
from models import Stereo
from utils import myUtils
from tensorboardX import SummaryWriter
import skimage
import skimage.io
import skimage.transform
from submission.Submission import Submission as Base
import collections


# Submission for any stereo model including SR-Stereo
class Submission(Base):
    def __init__(self, subImgLoader):
        super(Submission, self).__init__(subImgLoader)

    def _subIt(self, batch):
        dispOut = self.model.predict(*batch[0:2], mode='left')
        dispOut = dispOut.squeeze()
        dispOut = dispOut.data.cpu().numpy()
        dispOut = (dispOut * 256).astype('uint16')

        outputs = collections.OrderedDict()
        outputs['dispOutL'] = dispOut
        return outputs


def main():
    parser = myUtils.getBasicParser(
        ['maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'dataset', 'subtype'],
        description='generate png image for kitti final submission')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    import dataloader
    if args.subtype == 'eval':
        batchSizes = (0, 1)
    _, imgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                            batchSizes=batchSizes, mode='submission',
                                            mask=(1, 1, 0, 0))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, dispScale=args.dispscale, cuda=args.cuda, stage=stage)
    stereo.load(args.loadmodel)

    # Submission
    sub = Submission(subImgLoader=imgLoader)
    sub(model=stereo)


if __name__ == '__main__':
    main()
