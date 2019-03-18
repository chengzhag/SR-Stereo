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
        rawOutputs = self.model.predict(batch.detach(), mask=(1, 1))
        if myUtils.depth(rawOutputs) == 4:
            rawOutputs = rawOutputs[-1]

        outputs = collections.OrderedDict()
        for gtDisp, rawOutputsSide, side in zip(batch.lowestResDisps(), rawOutputs, ('L', 'R')):
            dispOut = myUtils.getLastNotList(rawOutputsSide)
            outputs['dispOut' + side] = myUtils.savePreprocessDisp(dispOut)
            if gtDisp is not None:
                outputs['gtDisp' + side] = myUtils.savePreprocessDisp(gtDisp)

            # for SRStereo, save outDispHighs together
            if type(rawOutputsSide) in (tuple, list) and len(rawOutputsSide) == 2 \
                and type(rawOutputsSide[1]) in (tuple, list) and len(rawOutputsSide[1]) == 2:
                outputs['dispOutHigh' + side] = myUtils.savePreprocessDisp(rawOutputsSide[1][0], dispScale=170)


        return outputs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'maxdisp', 'dispscale', 'model', 'datapath', 'loadmodel', 'no_cuda', 'dataset', 'subtype', 'load_scale', 'half'],
        description='generate png image for kitti final submission')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    import dataloader

    _, imgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                            batchSizes=(0, 1),
                                            loadScale=args.load_scale,
                                            mode=args.subtype,
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
