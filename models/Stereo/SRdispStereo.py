import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .SRStereo import SRStereo
from .. import SR
from .PSMNetDown import PSMNetDown


class SRdispStereo(SRStereo):
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRdispStereo, self).__init__(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half,
                                           stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._getSr = lambda: SR.SRdisp(cuda=cuda, half=half, stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()

        cated, warpTos = self._sr.warpAndCat(batch)
        batch.highResRGBs(cated)
        outputs = super(SRdispStereo, self).predict(batch, mask)
        outputsReturn = [[warpTo] + outputsSide for warpTo, outputsSide in zip(warpTos, outputs)]
        return outputsReturn

    def test(self, batch, evalType='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, (4, 8))
        if len(batch) == 8:
            batch = batch.lastScaleBatch()

        scores, outputs, rawOutputs = super(SRdispStereo, self).test(batch, evalType, returnOutputs, kitti)
        for (warpTo, outSR, (outDispHigh, outDispLow)), side in zip(rawOutputs, ('L', 'R')):
            if returnOutputs:
                if warpTo is not None:
                    outputs['warpTo' + side] = warpTo
        return scores, outputs, rawOutputs

