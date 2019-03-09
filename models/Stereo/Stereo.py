import torch
from evaluation import evalFcn
from utils import myUtils
from ..Model import Model
import collections


class Stereo(Model):
    # dataset: only used for suffix of saveFolderName
    # maxdisp: disparity range of self.model.
    # (
    # Note: the definition of maxdisp was changed from commit:d46b96.
    # This corrects loss computation. So there will be different behavious in Stereo_train_moduletest.
    # Change 'gts < self.maxdisp' to 'gts < self.outputMaxDisp' will reproduce the original wrong loss curve.
    # )
    # dispScale: scale the disparity value before input the original disparity map
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(Stereo, self).__init__(cuda, half, stage, dataset, saveFolderSuffix)
        self.maxdisp = maxdisp
        self.dispScale = dispScale
        self.outputMaxDisp = maxdisp * dispScale  # final output value range of disparity map

    def predict(self, batch, mask=(1,1)):
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()

        imgL, imgR = batch.lowestResRGBs()

        with torch.no_grad():
            outputs = []
            for inputL, inputR, process, do in zip((imgL, imgR), (imgR, imgL),
                                                   (lambda im: list(im) if type(im) is tuple else im, myUtils.flipLR), mask):
                if do:
                    output = process(self.model(process(inputL), process(inputR)))
                    outputs.append(output)
                else:
                    outputs.append(None)

            return outputs

    def test(self, batch, evalType='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, 4)

        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        mask = [disp is not None for disp in disps]
        rawOutputs = self.predict(batch, mask)

        for gt, rawOutputsSide, side in zip(disps, rawOutputs, ('L', 'R')):
            dispOut = myUtils.getLastNotList(rawOutputsSide)
            if dispOut is not None:
                if returnOutputs:
                    outputs['outputDisp' + side] = dispOut / self.outputMaxDisp

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

                scores[evalType + side] = evalFcn.getEvalFcn(evalType)(gt, dispOut)

        return scores, outputs, rawOutputs

    def save(self, epoch, iteration, trainLoss, additionalInfo=None):
        additionalInfo = {} if additionalInfo is None else additionalInfo
        additionalInfo.update({
            'maxdisp': self.maxdisp,
            'dispScale': self.dispScale,
            'outputMaxDisp': self.outputMaxDisp
        })
        super(Stereo, self).save(epoch, iteration, trainLoss, additionalInfo)


