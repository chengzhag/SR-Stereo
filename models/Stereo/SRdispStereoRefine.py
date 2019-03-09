import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .SRdispStereo import SRdispStereo
from .. import SR
import torch.nn.functional as F
from evaluation import evalFcn



class SRdispStereoRefine(SRdispStereo):
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRdispStereoRefine, self).__init__(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half,
                                           stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self.itRefine=0

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
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
            outputsReturn = [[[outSRs, psmnetDownOut] for psmnetDownOut in psmnetDownOuts]]
            if self.itRefine > 0:
                initialDisps = [myUtils.getLastNotList(dispsSide).unsqueeze(1).type_as(batch[0]) for dispsSide in psmnetDownOuts]
                batch.lowestResDisps(initialDisps)
                for i in range(self.itRefine):
                    itOutputs = super(SRdispStereoRefine, self).predict(batch.detach(), mask=mask)
                    outputsReturn.append(itOutputs)
                    dispOuts = [myUtils.getLastNotList(itOutputsSide).unsqueeze(1).type_as(batch[0])
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
            for gtDisp, gtSR, rawOutputSide, side, iSide in zip(disps, gtSRs, rawOutput, ('L', 'R'), (0, 1)):
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
                if gtSR is not None:
                    scores['l1' + 'Sr' + side + itSuffix] = evalFcn.l1(gtSR, outSRs[iSide])

                if dispOut is not None:
                    if returnOutputs:
                        outputs['outputDispLow' + side + itSuffix] = dispOut / self.outputMaxDisp

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

                    scores[evalType + side + itSuffix] = evalFcn.getEvalFcn(evalType)(gtDisp, dispOut)

        return scores, outputs, rawOutputs
