from utils import myUtils
import argparse
import time
import torch
import os
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
        self.model = None

    def _subIt(self, batch):
        pass

    def __call__(self, model):
        self.model = model
        saveFolder = os.path.join(self.model.checkpointFolder, 'Submission')
        myUtils.checkDir(saveFolder)
        tic = time.time()
        ticFull = time.time()
        for iIm, batch in enumerate(self.subImgLoader, 1):
            batch = [data if data.numel() else None for data in batch]
            name = self.subImgLoader.dataset.name(iIm - 1)
            name, extension = os.path.splitext(name)

            outputs = self._subIt(batch)

            for folder, im in outputs.items():
                myUtils.checkDir(os.path.join(saveFolder, folder))
                saveDir = os.path.join(saveFolder, folder, name + '.png')
                skimage.io.imsave(saveDir, im)
                print('saving to: %s' % saveDir)

            timeLeft = (time.time() - tic) / 60 * (len(self.subImgLoader) - iIm)
            print('im %d/%d, left %.2fmin' % (
                iIm, len(self.subImgLoader),
                timeLeft))
            tic = time.time()
        submissionTime = time.time() - ticFull
        print('Full submission time = %.2fmin' % (submissionTime / 60))
