import os
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from evaluation import evalFcn
from utils import myUtils
from apex import amp

class Model:
    def __init__(self, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        self.cuda = cuda
        self.half = half
        self.amp_handle = amp.init(half)
        self.stage = stage

        self.startTime = time.localtime(time.time())
        self.multiple = 16

        self.saveFolderName = time.strftime('%y%m%d%H%M%S_', self.startTime) \
                              + self.__class__.__name__ \
                              + saveFolderSuffix
        if dataset is not None: self.saveFolderName += ('_%s' % dataset)
        self.saveFolder = os.path.join('logs', stage, self.saveFolderName)
        self.logFolder = None
        self.checkpointDir = None
        self.checkpointFolder = None

        self.getModel = None
        self.model = None
        self.optimizer = None

    def initModel(self):
        pass

    def trainPrepare(self, batch=()):
        if self.model is None:
            self.initModel()
        # When training, log files should be saved to saveFolder.
        self.logFolder = os.path.join(self.saveFolder, 'logs')
        self.model.train()
        return batch

    def loss(self, outputs, gts, weights=(1,)):
        raise Exception('Error: please overtide \'Model.loss()\' without calling it!')

    def train(self, batch):
        raise Exception('Error: please overtide \'Model.train()\' without calling it!')

    def predictPrepare(self, batch=()):
        self.model.eval()
        return batch

    def predict(self, batch):
        raise Exception('Error: please overtide \'Model.predict()\' without calling it!')

    def load(self, checkpointDir):
        if checkpointDir is not None:
            print('Loading checkpoint from %s' % checkpointDir)
        else:
            raise Exception('checkpoint dir is None!')

        # update checkpointDir
        self.checkpointDir = checkpointDir
        self.checkpointFolder, _ = os.path.split(self.checkpointDir)
        # When testing, log files should be saved to checkpointFolder.
        # Here checkpointFolder is setted as default logging folder.
        self.logFolder = os.path.join(self.checkpointFolder, 'logs')

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def beforeSave(self, epoch, iteration):
        # update checkpointDir
        self.checkpointDir = os.path.join(self.saveFolder, 'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        self.checkpointFolder = self.saveFolder
        self.logFolder = os.path.join(self.checkpointFolder, 'logs')
        myUtils.checkDir(self.saveFolder)
