import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .Stereo import Stereo
from .. import SR
from .PSMNetDown import PSMNetDown

class RawSRStereo(nn.Module):
    def __init__(self, stereo, sr):
        super(RawSRStereo, self).__init__()
        self.stereo = stereo
        self.sr = sr

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale / 2
    def forward(self, left, right, updateSR=True):
        if updateSR:
            outSrL = self.sr.forward(left)
            outSrR = self.sr.forward(right)
        else:
            with torch.no_grad():
                outSrL = self.sr.forward(left).detach()
                outSrR = self.sr.forward(right).detach()
        outDispHighs, outDispLows = self.stereo.forward(outSrL, outSrR)
        return (outSrL, outSrR), (outDispHighs, outDispLows)

class SRStereo(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRStereo, self).__init__(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half,
                                       stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._getSr = lambda: SR.SR(cuda=cuda, half=half, stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._getStereo = lambda: PSMNetDown(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half, stage=stage,
                                  dataset=dataset,
                                  saveFolderSuffix=saveFolderSuffix)
        self.getModel = RawSRStereo
        self._stereo = None
        self._sr = None

    def initModel(self):
        self._stereo = self._getStereo()
        self._stereo.initModel()
        self._stereo.optimizer = None
        self._sr = self._getSr()
        self._sr.initModel()
        self._sr.optimizer = None
        self.model = self.getModel(self._stereo.model, self._sr.model)
        self.outputMaxDisp = self._stereo.outputMaxDisp

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def test(self, batch, evalType='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, (4, 8))
        if len(batch) == 8:
            batch = batch.lastScaleBatch()

        scores, outputs, rawOutputs = super(SRStereo, self).test(batch, evalType, returnOutputs, kitti)
        for rawOutputsSide, side in zip(rawOutputs, ('L', 'R')):
            if rawOutputsSide is not None:
                outSRs, (outDispHigh, outDispLow) = rawOutputsSide[-2:]
                if returnOutputs:
                    if outDispHigh is not None:
                        outputs['outputDispHigh' + side] = outDispHigh / (self.outputMaxDisp * 2)
                    for outSr, sideSr in zip(outSRs, ('L', 'R')):
                        if outSr is not None:
                            outputs['outputSr' + sideSr + side] = outSr
        return scores, outputs, rawOutputs

    def loss(self, outputs, gts, kitti=False):
        losses = []
        (outSrL, outSrR), (outDispHighs, outDispLows) = outputs
        (srGtL, srGtR), (dispHighGTs, dispLowGTs), (inputL, inputR) = gts

        # get SR outputs loss
        lossSR = None

        for outSr, srGt, input in zip((outSrL, outSrR), (srGtL, srGtR), (inputL, inputR)):
            if all([t is not None for t in (outSr, srGt)]):
                if outSr.size() == srGt.size():
                    lossSRside = self._sr.loss(outSr, srGt)
                else:
                    lossSRside = self._sr.loss(nn.AvgPool2d((2, 2))(outSr), srGt)
            elif all([t is not None for t in (outSr, input)]):
                # if dataset has no SR GT, use lowestResRGBs as GTs
                lossSRside = self._sr.loss(nn.AvgPool2d((2, 2))(outSr), srGt)
            else:
                lossSRside = None
            lossSR = lossSRside if lossSR is None else lossSR + lossSRside

        if lossSR is not None:
            lossSR /= 2
        losses.append(lossSR)

        # get disparity losses
        lossDisps = self._stereo.loss((outDispHighs, outDispLows), (dispHighGTs, dispLowGTs), kitti=kitti)
        losses += lossDisps

        return losses

    def trainOneSide(self, inputL, inputR, srGtL, srGtR, dispGTs, returnOutputs=False, kitti=False,
                     weights=(0, 0, 1)):
        self.optimizer.zero_grad()

        doUpdateSR = weights[0] >= 0
        (outSrL, outSrR), (outDispHighs, outDispLows) = self.model.forward(inputL, inputR, updateSR=doUpdateSR)

        losses = self.loss(
            ((outSrL, outSrR) if doUpdateSR else (None, None), (outDispHighs, outDispLows)),
            ((srGtL, srGtR), dispGTs, (inputL, inputR)),
            kitti=kitti
        )
        loss = sum([weight * loss for weight, loss in zip(weights, losses) if loss is not None])
        with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

        if returnOutputs:
            with torch.no_grad():
                outputs = (myUtils.quantize(outSrL.detach(), 1),
                           myUtils.quantize(outSrR.detach(), 1),
                           outDispHighs[2].detach() / (self.outputMaxDisp * 2),
                           outDispLows[2].detach() / self.outputMaxDisp)
                outputs = [output.detach() for output in outputs]
        else:
            outputs = []

        losses = [loss] + losses
        return [loss.data.item() if hasattr(loss, 'data') else loss for loss in losses], outputs

    # weights: weights of
    #   SR output losses (lossSR),
    #   SR disparity map losses (lossDispHigh),
    #   normal sized disparity map losses (lossDispLow)
    def train(self, batch, returnOutputs=False, kitti=False, weights=(0, 1, 0)):
        myUtils.assertBatchLen(batch, (4, 8))
        self.trainPrepare()
        if len(batch) == 4:
            batch = myUtils.Batch([None] * 4 + batch.batch)

        imgLowL, imgLowR = batch.lowestResRGBs()
        imgHighL, imgHighR = batch.highResRGBs()

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        for inputL, inputR, srGtL, srGtR, dispGTs, process, side in zip(
                (imgLowL, imgLowR), (imgLowR, imgLowL),
                (imgHighL, imgHighR), (imgHighR, imgHighL),
                zip(batch.highResDisps(), batch.lowResDisps()),
                (lambda im: im, myUtils.flipLR),
                ('L', 'R')
        ):
            if (not all([gt is None for gt in dispGTs])) \
                    or (all([t is not None for t in (srGtL, srGtR)])):
                lossesList, outputsList = self.trainOneSide(
                    *process([inputL, inputR, srGtL, srGtR, dispGTs]),
                    returnOutputs=returnOutputs,
                    kitti=kitti,
                    weights=weights
                )
                for suffix, loss in zip(('', 'Sr', 'DispHigh', 'Disp'), lossesList):
                    if loss is not None:
                        losses['loss' + suffix + side] = loss

                if returnOutputs:
                    suffixSR = ('SrL', 'SrR') if side == 'L' else ('SrR', 'SrL')
                    for suffix, output in zip(suffixSR + ('DispHigh', 'DispLow'), outputsList):
                        outputs['output' + suffix + side] = process(output)

        return losses, outputs

    def load(self, checkpointDir):
        if checkpointDir is None:
            return None, None
        checkpointDir = super(SRStereo, self).loadPrepare(checkpointDir, 2)

        if type(checkpointDir) in (list, tuple) and len(checkpointDir) == 2:
            # Load pretrained SR and Stereo weights
            self._sr.load(checkpointDir[0])
            self._stereo.load(checkpointDir[1])
            return None, None
        elif type(checkpointDir) is str:
            # Load fintuned SRStereo weights
            return super(SRStereo, self).load(checkpointDir)
        else:
            raise Exception('Error: SRStereo need 2 checkpoints SR/Stereo or 1 checkpoint SRStereo to load!')




