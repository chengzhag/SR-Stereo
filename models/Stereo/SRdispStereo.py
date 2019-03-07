import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .Stereo import Stereo
from .. import SR
from .PSMNetDown import PSMNetDown
from models.SR.warp import warpAndCat

class RawSRdispStereo(nn.Module):
    def __init__(self, stereo, sr):
        super(RawSRdispStereo, self).__init__()
        self.stereo = stereo
        self.sr = sr

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale / 2
    def forward(self, left, right, dispL, dispR, updateSR=True):
        with torch.no_grad():
            cated, warpTos = warpAndCat([left, right, dispL, dispR], False)
            cated, warpTos = [c.detach() for c in cated], [w.detach() for w in warpTos]
        if updateSR:
            outSrL = self.sr.forward(cated[0])
            outSrR = self.sr.forward(cated[1])
        else:
            with torch.no_grad():
                outSrL = self.sr.forward(cated[0]).detach()
                outSrR = self.sr.forward(cated[1]).detach()
        outDispHighs, outDispLows = self.stereo.forward(outSrL, outSrR)
        return warpTos, (outSrL, outSrR), (outDispHighs, outDispLows)

class SRdispStereo(Stereo):
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(SRdispStereo, self).__init__(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half,
                                           stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._sr = SR.SRdisp(cuda=cuda, half=half, stage=stage, dataset=dataset, saveFolderSuffix=saveFolderSuffix)
        self._stereo = PSMNetDown(maxdisp=maxdisp, dispScale=dispScale, cuda=cuda, half=half, stage=stage,
                                  dataset=dataset,
                                  saveFolderSuffix=saveFolderSuffix)
        self.outputMaxDisp = self._stereo.outputMaxDisp
        self.getModel = RawSRdispStereo

    def initModel(self):
        self._stereo.initModel()
        self._stereo.optimizer = None
        self._sr.initModel()
        self._sr.optimizer = None
        self.model = self.getModel(self._stereo.model, self._sr.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))



