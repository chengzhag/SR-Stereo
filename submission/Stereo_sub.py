import torch
import os
from models import Stereo
from utils import myUtils
from submission.Submission import Submission as Base
import collections


# Submission for any stereo model including SR-Stereo
class Submission(Base):
    def __init__(self, subImgLoader):
        super(Submission, self).__init__(subImgLoader)

    def _subIt(self, batch):
        rawOutputs = self.model.predict(batch.detach(), mask=(1, 0))[0]
        dispOut = myUtils.getLastNotList(rawOutputs)

        outputs = collections.OrderedDict()
        outputs['dispOutL'] = myUtils.savePreprocessDisp(dispOut)
        gtL = batch.highResDisps()[0]
        if gtL is not None:
            outputs['gtL'] = myUtils.savePreprocessDisp(gtL)
        return outputs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'dataset', 'subtype', 'load_scale', 'half'],
        description='generate png image for kitti final submission')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    import dataloader
    if args.subtype == 'eval':
        batchSizes = (0, 1)
    _, imgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                            batchSizes=batchSizes,
                                            loadScale=args.load_scale,
                                            mode='submission',
                                            mask=(1, 1, 1, 0))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stage = os.path.join(args.outputFolder, stage) if args.outputFolder is not None else stage
    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, dispScale=args.dispscale, 
                                         cuda=args.cuda, half=args.half, stage=stage)
    stereo.load(args.loadmodel)

    # Submission
    sub = Submission(subImgLoader=imgLoader)
    sub(model=stereo)


if __name__ == '__main__':
    main()
