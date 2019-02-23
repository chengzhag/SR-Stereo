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


# Submission for any stereo model including SR-Stereo
class Submission:
    def __init__(self, subImgLoader):
        if max(subImgLoader.batchSizes) > 1:
            raise Exception('subImgLoader for Submission can only have batchSize equal to 1!')
        self.subImgLoader = subImgLoader
        self.stereo = None

    def __call__(self, stereo):
        self.stereo = stereo
        saveFolder = os.path.join(self.stereo.checkpointFolder, 'Stereo_sub')
        myUtils.checkDir(saveFolder)
        tic = time.time()
        ticFull = time.time()
        for iIm, ims in enumerate(self.subImgLoader, 1):
            nameL = self.subImgLoader.dataset.inputLdirs[iIm - 1].split('/')[-1]
            savePath = os.path.join(saveFolder, nameL)
            ims = [data if data.numel() else None for data in ims]
            dispOut = self.stereo.predict(*ims[0:2], mode='left')
            dispOut = dispOut.squeeze()
            dispOut = dispOut.data.cpu().numpy()
            skimage.io.imsave(savePath, (dispOut * 256).astype('uint16'))

            timeLeft = (time.time() - tic) / 60 * (len(self.subImgLoader) - iIm)
            print('im %d/%d, %s, left %.2fmin' % (
                iIm, len(self.subImgLoader),
                savePath, timeLeft))
            tic = time.time()
        submissionTime = time.time() - ticFull
        print('Full submission time = %.2fmin' % (submissionTime / 60))


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
    sub(stereo=stereo)


if __name__ == '__main__':
    main()
