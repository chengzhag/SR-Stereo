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
        for iIm, ims in enumerate(self.subImgLoader):
            nameL = self.subImgLoader.dataset.inputLdirs[iIm].split('/')[-1]
            ims = [data if data.numel() else None for data in ims]
            dispOut = self.stereo.predict(*ims[0:2], mode='left')
            dispOut = dispOut.squeeze()
            dispOut = dispOut.data.cpu().numpy()
            skimage.io.imsave(os.path.join(saveFolder, nameL), (dispOut * 256).astype('uint16'))



def main():
    parser = argparse.ArgumentParser(description='Stereo')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='PSMNet',
                        help='select model')
    parser.add_argument('--datapath', default='../datasets/sceneflow/',
                        help='datapath')
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dataset', type=str, default='kitti2015',
                        help='(sceneflow/kitti2012/kitti2015/carla_kitti)')
    parser.add_argument('--subtype', type=str, default='eval',
                        help='dataset type used for submission (eval/test)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    import dataloader
    if args.subtype == 'eval':
        batchSizes = (0, 1)
    _, imgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                batchSizes=batchSizes, mode='submission')

    # Load model
    stage, _ = os.path.splitext(os.path.basename(__file__))
    stereo = getattr(Stereo, args.model)(loadScale=imgLoader.loadScale, cropScale=imgLoader.cropScale,
                                         maxdisp=args.maxdisp, cuda=args.cuda, stage=stage)
    stereo.load(args.loadmodel)

    # Submission
    sub = Submission(subImgLoader=imgLoader)
    sub(stereo=stereo)

if __name__ == '__main__':
    main()

