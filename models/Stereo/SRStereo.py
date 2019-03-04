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
    def forward(self, left, right):
        outSrL = self.sr.forward(left)
        outSrR = self.sr.forward(right)
        outDispHighs, outDispLows = self.stereo.forward(outSrL, outSrR)
        return (outSrL, outSrR), (outDispHighs, outDispLows)

class SRStereo(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRStereo, self).__init__(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half,
                                       stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._sr = SR.SR(cuda=cuda, half=half, stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._stereo = PSMNetDown(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half, stage=stage,
                                  dataset=dataset,
                                  saveFolderSuffix=saveFolderSuffix)
        self.outputMaxDisp = self._stereo.outputMaxDisp
        self.getModel = RawSRStereo

    def initModel(self):
        self._stereo.initModel()
        self._stereo.optimizer = None
        self._sr.initModel()
        self._sr.optimizer = None
        self.model = self.getModel(self._stereo.model, self._sr.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def predict(self, batch, mask=(1, 1)):
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()

        # One method to predict
        # srs = self._sr.predict(batch)
        # batch = myUtils.Batch(4)
        # batch.highResRGBs(srs)
        # outputs = self._stereo.predict(batch, mask=mask)

        # Another method to predict which can test forward fcn
        outputs = super(SRStereo, self).predict(batch, mask)
        outputs = [output[1][1] for output in outputs]
        return outputs

    def test(self, batch, type='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, 4)

        # Test with outputing sr images
        # srs = self._sr.predict(batch)
        #
        # stereoBatch = myUtils.Batch(8)
        # stereoBatch.highResRGBs(srs)
        # stereoBatch.lowestResDisps(batch.lowestResDisps())
        # scores, outputs = self._stereo.test(stereoBatch, type=type, returnOutputs=returnOutputs, kitti=kitti)
        #
        # for sr, side in zip(srs, ('L', 'R')):
        #     outputs['outputSr' + side] = sr
        # return scores, outputs

        # Test without outputing sr images
        return super(SRStereo, self).test(batch, type=type, returnOutputs=returnOutputs, kitti=kitti)

    def loss(self, outputs, gts, kitti=False):
        losses = []
        (outSrL, outSrR), (outDispHighs, outDispLows) = outputs
        (srGtL, srGtR), (dispHighGTs, dispLowGTs) = gts

        # get SR outputs loss
        lossSR = 0
        if all([t is not None for t in (srGtL, srGtR)]) :
            for outSr, srGt in zip((outSrL, outSrR), (srGtL, srGtR)):
                lossSR += self._sr.loss(outSr, srGt)
        losses.append(lossSR)

        # get disparity losses
        lossDisps = self._stereo.loss((outDispHighs, outDispLows), (dispHighGTs, dispLowGTs), kitti=kitti)
        losses += lossDisps

        return losses

    def trainOneSide(self, inputL, inputR, srGtL, srGtR, dispGTs, returnOutputs=False, kitti=False,
                     weights=(0, 0, 1)):
        self.optimizer.zero_grad()
        (outSrL, outSrR), (outDispHighs, outDispLows) = self.model.forward(inputL, inputR)
        losses = self.loss(
            ((outSrL, outSrR), (outDispHighs, outDispLows)),
            ((srGtL, srGtR), dispGTs),
            kitti=kitti
        )

        loss = sum([weight * loss for weight, loss in zip(weights, losses)])
        with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

        if returnOutputs:
            with torch.no_grad():
                outputs = (myUtils.quantize(outSrL, 1),
                           myUtils.quantize(outSrR, 1),
                           outDispHighs[2] / (self.outputMaxDisp * 2),
                           outDispLows[2] / self.outputMaxDisp)
                outputs = [output.detach() for output in outputs]
        else:
            outputs = []

        losses = [loss] + losses
        return [loss.data.item() for loss in losses], outputs

    # weights: weights of
    #   SR output losses (lossSR),
    #   SR disparity map losses (lossDispHigh),
    #   normal sized disparity map losses (lossDispLow)
    def train(self, batch, returnOutputs=False, kitti=False, weights=(0, 1, 0)):
        myUtils.assertBatchLen(batch, 8)
        self.trainPrepare()
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
                for suffix, loss in zip(('', 'Sr', 'DispHigh', 'DispLow'), lossesList):
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



