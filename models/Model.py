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

    def loss(self, outputs, gts):
        raise Exception('Error: please overtide \'Model.loss()\' without calling it!')

    def train(self, batch):
        raise Exception('Error: please overtide \'Model.train()\' without calling it!')

    def predictPrepare(self, batch=()):
        self.model.eval()
        return batch

    def predict(self, batch):
        raise Exception('Error: please overtide \'Model.predict()\' without calling it!')

    def beforeLoad(self, checkpointDirs):
        if checkpointDirs is not None:
            print('Loading checkpoint from %s' % checkpointDirs)
        else:
            print('No checkpoint specified. Will initialize weights randomly.')
            return None

        if type(checkpointDirs) in (list, tuple):
            if len(checkpointDirs) == 1:
                checkpointDirs = checkpointDirs[0]
                # update checkpointDir
                self.checkpointDir = checkpointDirs
                self.checkpointFolder, _ = os.path.split(self.checkpointDir)

            elif len(checkpointDirs) >= 1:
                # for model composed with multiple models, check if checkpointDirs are together
                modelRoot = None
                for dir in checkpointDirs:
                    checkpointFolder, _ = os.path.split(dir)
                    checkpointRoot = os.path.join(*checkpointFolder.split('/')[:-2])
                    if modelRoot is None:
                        modelRoot = checkpointRoot
                    elif modelRoot != checkpointRoot:
                        raise Exception('Error: For good project structure, '
                                        'checkpoints of model combinations should be placed together like: '
                                        'pycharmruns (running stage)/SRStereo_eval_test (model)/SR_train (components)/'
                                        '190228011913_SR_loadScale_10_trainCrop_96_1360_batchSize_4_carla_kitti (runs)/'
                                        '*.tar (checkpoints)')
                self.checkpointFolder = self.saveFolder

        # When testing, log files should be saved to checkpointFolder.
        # Here checkpointFolder is setted as default logging folder.
        self.logFolder = os.path.join(self.saveFolder, 'logs')

        return checkpointDirs

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def beforeSave(self, epoch, iteration):
        # update checkpointDir
        self.checkpointDir = os.path.join(self.saveFolder, 'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        self.checkpointFolder = self.saveFolder
        self.logFolder = os.path.join(self.checkpointFolder, 'logs')
        myUtils.checkDir(self.saveFolder)
