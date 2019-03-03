import torch
from utils import myUtils
from models.SR.warp import warp
from .SR import SR


class SRdisp(SR):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, cInput=6, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SRdisp, self).__init__(cInput, cuda, half, stage, dataset, saveFolderSuffix)

    def initModel(self):
        super(SRdisp, self).initModel()

    def warpAndCat(self, batch):
        inputL, inputR, dispL, dispR = batch
        with torch.no_grad():
            warpToL, warpToR, maskL, maskR = warp(*batch)
            warpTos = (warpToL, warpToR)
            cated = []
            for input in zip((inputL, inputR), (warpToL, warpToR), (maskL, maskR)):
                if self.args.n_inputs == 7:
                    cated.append(torch.cat(input, 1))
                elif self.args.n_inputs == 6:
                    cated.append(torch.cat(input[:2], 1))
                else:
                    raise Exception(
                        'Error: self.model.args.n_inputs = %d which is not supporty!' % self.model.args.n_inputs)
            return cated, warpTos

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
        return outputs

