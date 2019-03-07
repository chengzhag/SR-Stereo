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
            outSrL = self.sr.forward(left)
            outSrR = self.sr.forward(right)
        else:
            with torch.no_grad():
                outSrL = self.sr.forward(left).detach()
                outSrR = self.sr.forward(right).detach()
        outDispHighs, outDispLows = self.stereo.forward(outSrL, outSrR)
        return (outSrL, outSrR), (outDispHighs, outDispLows)

