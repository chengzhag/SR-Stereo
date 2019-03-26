import torch
import os
from models import SR
from utils import myUtils
from submission.Submission import Submission as Base
import collections


# Submission for any stereo model including SR-Stereo
class Submission(Base):
    def __init__(self, subImgLoader):
        super(Submission, self).__init__(subImgLoader)

    def _subIt(self, batch):
        outputs = collections.OrderedDict()
        if len(batch) == 8:
            dispsSR = batch.highResDisps()
            gts = batch.highResRGBs()
            for dispSr, gt, suffix in zip(dispsSR, gts, ('L', 'R')):
                if dispSr is not None:
                    outputs['dispSr' + suffix] = myUtils.savePreprocessDisp(dispSr)
                if gt is not None:
                    outputs['gtSr' + suffix] = myUtils.savePreprocessRGB(gt)
            batch = batch.lastScaleBatch()

        outSRs = self.model.predict(batch=batch.lastScaleBatch().detach())

        inputs = batch.lowestResRGBs()

        for input, outSR, suffix in zip(inputs, outSRs, ('L', 'R')):
            if input is not None:
                outputs['outputSr' + suffix] = myUtils.savePreprocessRGB(outSR)
                outputs['input' + suffix] = myUtils.savePreprocessRGB(input)

        return outputs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'datapath', 'loadmodel', 'no_cuda', 'dataset', 'subtype', 'load_scale', 'half'],
        description='generate png image for SR submission')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    import dataloader

    _, imgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                            batchSizes=(0, 1),
                                            loadScale=args.load_scale,
                                            mode=args.subtype,
                                            mask=(1, 1, 1, 1))

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stage = os.path.join(args.outputFolder, stage) if args.outputFolder is not None else stage
    sr = getattr(SR, 'SR')(cuda=args.cuda, half=args.half, stage=stage,
                           dataset=args.dataset)
    if args.loadmodel is not None:
        sr.load(args.loadmodel)

    # Submission
    sub = Submission(subImgLoader=imgLoader)
    sub(model=sr)


if __name__ == '__main__':
    main()
