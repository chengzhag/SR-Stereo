from utils import myUtils
from ..Model import Model
from .. import SR
from .PSMNetDown import PSMNetDown


class SRStereo(Model):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRStereo, self).__init__(cuda, half, stage, dataset, saveFolderSuffix)
        self.sr = SR.SR(cuda=cuda, half=half, stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self.stereo = PSMNetDown(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half, stage=stage,
                                 dataset=dataset,
                                 saveFolderSuffix=saveFolderSuffix)
        self.outputMaxDisp = self.stereo.outputMaxDisp

    def initModel(self):
        self.stereo.initModel()
        self.sr.initModel()

    def predict(self, batch, mask=(1, 1)):
        myUtils.assertBatchLen(batch, 4)
        srs = self.sr.predict(batch)
        batch = myUtils.Batch(4).highResRGBs(srs)
        outputs = self.stereo.predict(batch, mask=mask)
        return outputs

    def test(self, batch, type='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, 4)
        srs = self.sr.predict(batch)
        stereoBatch = myUtils.Batch(8)
        stereoBatch.highResRGBs(srs)
        stereoBatch.lowestResDisps(batch.lowestResDisps())
        scores, outputs = self.stereo.test(stereoBatch, type=type, returnOutputs=returnOutputs, kitti=kitti)
        for sr, side in zip(srs, ('L', 'R')):
            outputs['outputSR' + side] = sr
        return scores, outputs

    def load(self, checkpointDir):
        checkpointDir = self.loadPrepare(checkpointDir, 3)
        if type(checkpointDir) not in (list, tuple) or len(checkpointDir) != 2:
            raise Exception('Error: SRStereo need two checkpoints (SR/Stereo) to load!')
        self.sr.load(checkpointDir[0])
        self.stereo.load(checkpointDir[1])


