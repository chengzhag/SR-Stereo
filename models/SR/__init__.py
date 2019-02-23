import os
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from evaluation import evalFcn
from utils import myUtils
from .EDSR import edsr
from ..Model import Model
import torch.nn.parallel as P


class SR(Model):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, cuda=True, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SR, self).__init__(cuda, stage, dataset, saveFolderSuffix)
        class Arg:
            def __init__(self):
                self.n_resblocks = 16
                self.n_feats = 64
                self.scale = [2]
                self.rgb_range = 255
                self.n_colors = 3
                self.res_scale = 1
        self.args = Arg()
        self.initModel()

    def initModel(self):
        self.model = edsr.make_model(self.args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model.cuda()

    def train(self):
        super(SR, self)._train()
        # TODO

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, imgL):
        super(SR, self)._predict()
        with torch.no_grad():
            output =  P.data_parallel(self.model, imgL * self.args.rgb_range)
            output = myUtils.quantize(output, self.args.rgb_range) / self.args.rgb_range
            return output

    def test(self, imgL, imgH, type='l1'):
        if self.cuda:
            imgL, imgH = imgL.cuda(), imgH.cuda()

        output = self.predict(imgL)
        score = getattr(evalFcn, type)(imgH, output)

        return score, output

    def load(self, checkpointDir):
        super(SR, self).load(checkpointDir)

        state_dict = torch.load(checkpointDir)
        self.model.load_state_dict(state_dict, strict=False)

        print('Loading complete! Number of model parameters: %d' % self.nParams())


