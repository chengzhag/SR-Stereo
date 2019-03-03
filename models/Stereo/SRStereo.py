import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
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
        left = self.sr.forward(left)
        right = self.sr.forward(right)
        outputs = self.stereo.forward(left, right)
        return outputs, (left, right)

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
        outputs = [output[0][1] for output in outputs]
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




