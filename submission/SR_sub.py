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
        def preprocess(im):
            output = im.squeeze()
            output = output.data.cpu().numpy()
            output = output.transpose(1, 2, 0)
            output = (output * 255).astype('uint8')
            return output

        outSRs = self.model.predict(batch=batch.lastScaleBatch().detach())

        inputs = batch.lowestResRGBs()
        if len(batch) == 8:
            gts = batch.highResRGBs()
        else:
            gts = (None, None)
        outputs = collections.OrderedDict()
        for input, outSR, gt, suffix in zip(inputs, outSRs, gts, ('L', 'R')):
            if input is not None:
                outputs['outputSr' + suffix] = preprocess(outSR)
                outputs['input' + suffix] = preprocess(input)
                if gt is not None:
                    outputs['gtSr' + suffix] = preprocess(gt)

        return outputs


def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'datapath', 'loadmodel', 'no_cuda', 'dataset', 'subtype', 'load_scale', 'half'],
        description='generate png image for SR submission')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    import dataloader
    if args.subtype == 'eval':
        batchSizes = (0, 1)
    _, imgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                            batchSizes=batchSizes,
                                            loadScale=(args.load_scale[0]),
                                            mode='submission',
                                            mask=(1, 1, 0, 0))

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
