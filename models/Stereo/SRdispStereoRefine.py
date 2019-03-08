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

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1), it=0):
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
            if it > 0:
                initialDisps = [myUtils.getLastNotList(disps).unsqueeze(1).type_as(batch[0]) for disps in psmnetDownOuts]
                batch.lowestResDisps(initialDisps)
                for i in range(it):
                    itOutputs = super(SRdispStereoRefine, self).predict(batch.detach(), mask=mask)
                    outputsReturn.append(itOutputs)
                    dispOuts = [myUtils.getLastNotList(itOutputsSide) for itOutputsSide in itOutputs]
                    batch.lowestResDisps(dispOuts)

            return outputsReturn

    def test(self, batch, evalType='l1', returnOutputs=False, kitti=False, it=2):
        myUtils.assertBatchLen(batch, (4, 8))
        if len(batch) == 8:
            batch = batch.lastScaleBatch()

        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        mask = [disp is not None for disp in disps]
        rawOutputs = self.predict(batch, mask, it=it)

        for it, rawOutput in enumerate(rawOutputs):
            itSuffix = str(it)
            for gt, rawOutputSide, side in zip(disps, rawOutput, ('L', 'R')):
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

                if dispOut is not None:
                    if returnOutputs:
                        outputs['output' + side + itSuffix] = dispOut / self.outputMaxDisp

                    if dispOut.dim() == 2:
                        dispOut = dispOut.unsqueeze(0)
                    if dispOut.dim() == 3:
                        dispOut = dispOut.unsqueeze(1)

                    # for kitti dataset, only consider loss of none zero disparity pixels in gt
                    if kitti and evalType != 'outlierPSMNet':
                        mask = gt > 0
                        dispOut = dispOut[mask]
                        gt = gt[mask]
                    elif not kitti:
                        mask = gt < self.outputMaxDisp
                        dispOut = dispOut[mask]
                        gt = gt[mask]

                    scores[evalType + side + itSuffix] = evalFcn.getEvalFcn(evalType)(gt, dispOut)

        return scores, outputs, rawOutputs
