import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .SRdispStereo import SRdispStereo
from .. import SR
import torch.nn.functional as F
from evaluation import evalFcn
import random


class SRdispStereoRefine(SRdispStereo):
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRdispStereoRefine, self).__init__(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half,
                                           stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self.itRefine=0

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    # mask: useless in this case
    def predict(self, batch, mask=(1,1), itRefine=None):
        if itRefine is None:
            itRefine = self.itRefine
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()

        # initialize SR output from low res input
        with torch.no_grad():
            outSRs = [F.interpolate(
                lowResInput, scale_factor=2, mode='bilinear',align_corners=False
            ) for lowResInput in batch.lowestResRGBs()]
            initialBatch = myUtils.Batch(4)
            initialBatch.lowestResRGBs(outSRs)
            psmnetDownOuts = self._stereo.predict(initialBatch)
            outputsReturn = [[[outSRsReturn, psmnetDownOut]
                              for outSRsReturn, psmnetDownOut
                              in zip((outSRs, outSRs[::-1]), psmnetDownOuts)]]
            if itRefine > 0:
                initialDisps = [myUtils.getLastNotList(dispsSide).unsqueeze(1).type_as(batch[0])
                                if myUtils.getLastNotList(dispsSide) is not None else None
                                for dispsSide in psmnetDownOuts]
                batch.lowestResDisps(initialDisps)
                for i in range(itRefine):
                    itOutputs = super(SRdispStereoRefine, self).predict(batch.detach())
                    outputsReturn.append(itOutputs)
                    dispOuts = [myUtils.getLastNotList(itOutputsSide).unsqueeze(1).type_as(batch[0])
                                if myUtils.getLastNotList(itOutputsSide) is not None else None
                                for itOutputsSide in itOutputs]
                    batch.lowestResDisps(dispOuts)

            return outputsReturn

    def test(self, batch, evalType='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, (4, 8))
        if len(batch) == 8:
            gtSRs = batch.highResRGBs()
            batch = batch.lastScaleBatch()
        else:
            gtSRs = (None, None)

        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        mask = [disp is not None for disp in disps]
        rawOutputs = self.predict(batch, mask)

        for it, rawOutput in enumerate(rawOutputs):
            itSuffix = str(it)
            for gtDisp, gtSR, rawOutputSide, side in zip(disps, gtSRs, rawOutput, ('L', 'R')):
                if it == 0:
                    outSRs, (outDispHigh, dispOut) = rawOutputSide
                else:
                    warpTo, outSRs, (outDispHigh, dispOut) = rawOutputSide
                    if returnOutputs:
                        if warpTo is not None:
                            outputs['warpTo' + side + itSuffix] = warpTo
                if returnOutputs:
                    if outDispHigh is not None:
                        outputs['outputDispHigh' + side + itSuffix] = outDispHigh / (self.outputMaxDisp * 2)
                    for outSr, sideSr in zip(outSRs, ('L', 'R')):
                        if outSr is not None:
                            outputs['outputSr' + sideSr + side + itSuffix] = outSr
                if gtSR is not None and outSRs[0] is not None:
                    scoreSR = evalFcn.l1(
                        gtSR * self._sr.args.rgb_range, outSRs[0] * self._sr.args.rgb_range)
                    scores['l1' + 'Sr' + side + itSuffix] = scoreSR
                    if it == len(rawOutputs) - 1:
                        scores['l1' + 'Sr' + side] = scoreSR

                if dispOut is not None and gtDisp is not None:
                    if returnOutputs:
                        outputs['outputDisp' + side + itSuffix] = dispOut / self.outputMaxDisp

                    if dispOut.dim() == 2:
                        dispOut = dispOut.unsqueeze(0)
                    if dispOut.dim() == 3:
                        dispOut = dispOut.unsqueeze(1)

                    # for kitti dataset, only consider loss of none zero disparity pixels in gtDisp
                    if kitti and evalType != 'outlierPSMNet':
                        mask = gtDisp > 0
                        dispOut = dispOut[mask]
                        gtDisp = gtDisp[mask]
                    elif not kitti:
                        mask = gtDisp < self.outputMaxDisp
                        dispOut = dispOut[mask]
                        gtDisp = gtDisp[mask]

                    scoreDisp = evalFcn.getEvalFcn(evalType)(gtDisp, dispOut)
                    scores[evalType + side + itSuffix] = scoreDisp
                    if it == len(rawOutputs) - 1:
                        scores[evalType + side] = scoreDisp

        return scores, outputs, rawOutputs

    # weights: weights of
    #   SR output losses (lossSR),
    #   SR disparity map losses (lossDispHigh),
    #   normal sized disparity map losses (lossDispLow)
    def train(self, batch, returnOutputs=False, kitti=False, weights=(0, 1, 0), progress=0):
        myUtils.assertBatchLen(batch, (4, 8))
        if len(batch) == 4:
            batch = myUtils.Batch([None] * 4 + batch.batch, cuda=batch.cuda, half=batch.half)

        # if has no highResRGBs, use lowestResRGBs as GTs
        if all([sr is None for sr in batch.highResRGBs()]):
            batch.highResRGBs(batch.lowestResRGBs())

        # probability of training with dispsOut as input:
        # progress = [0, 1]: p = [0, 1]
        if random.random() < progress or kitti == True:
            if random.random() > progress:
                itRefine = random.randint(1, 2)
            else:
                itRefine = random.randint(0, 1)
            dispChoice = itRefine
            rawOuputs = self.predict(batch.lastScaleBatch(), mask=(1, 1), itRefine=itRefine)[-1]
            dispsOut = [myUtils.getLastNotList(rawOutputsSide).unsqueeze(1) for rawOutputsSide in rawOuputs]
            warpBatch = myUtils.Batch(batch.lowestResRGBs() + dispsOut, cuda=batch.cuda, half=batch.half)
        else:
            warpBatch = batch.lastScaleBatch()
            dispChoice = -1

        cated, warpTos = self._sr.warpAndCat(warpBatch)
        batch.lowestResRGBs(cated)

        losses, outputs = super(SRdispStereo, self).train(
            batch, returnOutputs=returnOutputs, kitti=kitti, weights=weights
        )
        losses['dispChoice'] = dispChoice
        for warpTo, side in zip(warpTos, ('L', 'R')):
            if returnOutputs:
                if warpTo is not None:
                    outputs['warpTo' + side] = warpTo

        return losses, outputs
