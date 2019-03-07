import torch
from utils import myUtils
from models.SR.warp import warpAndCat
from .SR import *

class SRdisp(SR):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SRdisp, self).__init__(cuda, half, stage, dataset, saveFolderSuffix)
        self.cInput = 6
        self.getModel = RawEDSR

    def withMask(self, doWith=True):
        if doWith:
            self.cInput = 7

    def warpAndCat(self, batch):
        return warpAndCat(batch.firstScaleBatch(), doCatMask=(self.cInput == 7))

    def initModel(self):
        super(SRdisp, self).initModel()

    def train(self, batch, returnOutputs=False):
        myUtils.assertBatchLen(batch, 8)
        self.trainPrepare()

        cated, warpTos = self.warpAndCat(batch.lastScaleBatch())
        batch.lowResRGBs(cated)
        losses, outputs = super(SRdisp, self).train(batch, returnOutputs)
        if returnOutputs:
            for warpTo, side in zip(warpTos, ('L', 'R')):
                outputs['warpTo' + side] = warpTo
        return losses, outputs

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()

        cated, warpTos = self.warpAndCat(batch.firstScaleBatch())
        batch.highResRGBs(cated)
        outputs = super(SRdisp, self).predict(batch, mask)
        return warpTos, outputs

    def test(self, batch, evalType='l1', returnOutputs=False):
        myUtils.assertBatchLen(batch, 8)

        scores, outputs, rawOutputs = super(SRdisp, self).test(batch, evalType, returnOutputs)
        warpTos, _ = rawOutputs
        if returnOutputs:
            for warpTo, side in zip(warpTos, ('L', 'R')):
                if warpTo is not None:
                    outputs['warpTo' + side] = warpTo
        return scores, outputs, rawOutputs
