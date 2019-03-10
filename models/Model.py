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

        self._newFolderName = time.strftime('%y%m%d%H%M%S_', self.startTime) \
                              + self.__class__.__name__ \
                              + saveFolderSuffix
        if dataset is not None:
            self._newFolderName += ('_%s' % dataset)
        self._newFolder = os.path.join('logs', stage, self._newFolderName)
        self.logFolder = None
        self.checkpointDir = None
        self.checkpointFolder = None

        self.getModel = None
        self.model = None
        self.optimizer = None

    def saveToNew(self):
        self.checkpointFolder = self._newFolder
        self.logFolder = os.path.join(self._newFolder, 'logs')

    def initModel(self):
        pass

    def trainPrepare(self):
        if type(self.model) in (list, tuple):
            for m in self.model:
                m.trainPrepare()
        else:
            if self.model is None:
                self.initModel()
            self.model.train()

    def loss(self, outputs, gts):
        raise Exception('Error: please overtide \'Model.loss()\' without calling it!')

    def train(self, batch):
        raise Exception('Error: please overtide \'Model.train()\' without calling it!')

    def predictPrepare(self):
        if type(self.model) in (list, tuple):
            for m in self.model:
                m.predictPrepare()
        else:
            self.model.eval()

    def predict(self, batch):
        raise Exception('Error: please overtide \'Model.predict()\' without calling it!')

    # For multi checkpointModel, optimizer state dict is saved to a independent folder.
    # If input checkpointDirs fewer than maxCheckPoints,
    #   optimizer checkpoint is not specified,
    #   new optimizer checkpoint will be saved to self.newFolder
    # If input checkpointDirs with amount of maxCheckPoints, optimizer checkpoint is the last one
    def loadPrepare(self, checkpointDirs, maxCheckPoints=1):
        if checkpointDirs is not None:
            print('Loading checkpoint from %s' % checkpointDirs)
        else:
            print('No checkpoint specified. Will initialize weights randomly.')
            return None

        if type(checkpointDirs) in (list, tuple):
            if len(checkpointDirs) == 1:
                checkpointDirs = checkpointDirs[0]

            elif len(checkpointDirs) >= 1:
                if len(checkpointDirs) > maxCheckPoints:
                    raise Exception(f'Error: Specified {len(checkpointDirs)} checkpoints. Only {maxCheckPoints} is(are) needed!')
                # # for model composed with multiple models, check if checkpointDirs are together
                # modelRoot = None
                # checkpointDirs = myUtils.scanCheckpoint(checkpointDirs)
                # for dir in checkpointDirs:
                #     checkpointFolder, _ = os.path.split(dir)
                #     checkpointRoot = os.path.join(*checkpointFolder.split('/')[:-2])
                #     if modelRoot is None:
                #         modelRoot = checkpointRoot
                #     elif modelRoot != checkpointRoot:
                #         raise Exception('Error: For good project structure, '
                #                         'checkpoints of model combinations should be placed together like: '
                #                         'pycharmruns (running stage)/SRStereo_eval_test (model)/SR_train (components)/'
                #                         '190228011913_SR_loadScale_10_trainCrop_96_1360_batchSize_4_carla_kitti (runs)/'
                #                         '*.tar (checkpoints)')
                # if len(checkpointDirs) == maxCheckPoints:
                #     self.checkpointDir = checkpointDirs[-1]

        if type(checkpointDirs) is str:
            checkpointDirs = myUtils.scanCheckpoint(checkpointDirs)

            # update checkpointDir
            self.checkpointDir = checkpointDirs

        if self.checkpointFolder is None and self.checkpointDir is not None:
            self.checkpointFolder, _ = os.path.split(self.checkpointDir)
            self.logFolder = os.path.join(self.checkpointFolder, 'logs')

        if self.model is None:
            self.initModel()
        return checkpointDirs

    def load(self, checkpointDir):
        checkpointDir = self.loadPrepare(checkpointDir)
        if checkpointDir is None:
            return None, None

        loadStateDict = torch.load(checkpointDir)

        loadModelDict = loadStateDict.get('state_dict', loadStateDict)
        try:
            match = self.model.load_state_dict(loadModelDict)
        except RuntimeError:
            match = self.model.module.load_state_dict(loadModelDict)

        if match:
            if 'optimizer' in loadStateDict.keys() and self.optimizer is not None:
                self.optimizer.load_state_dict(loadStateDict['optimizer'])
        else:
            print('Warning: Checkpoint dosent completely match current model. Optimizer will not be loaded!')
        print('Loading complete! Number of model parameters: %d' % self.nParams())

        epoch = loadStateDict.get('epoch')
        iteration = loadStateDict.get('iteration')
        print(f'Checkpoint epoch {epoch}, iteration {iteration}')
        return epoch, iteration

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def savePrepare(self, epoch, iteration):
        # update checkpointDir
        self.checkpointDir = os.path.join(self.checkpointFolder,
                                          'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        myUtils.checkDir(self.checkpointFolder)
        print('Saving model to: ' + self.checkpointDir)

    def save(self, epoch, iteration, trainLoss, additionalInfo=None):
        self.savePrepare(epoch, iteration)
        saveDict = {
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
        }
        if additionalInfo is not None:
            saveDict.update(additionalInfo)
        if self.optimizer is not None:
            saveDict['optimizer'] = self.optimizer.state_dict()
        torch.save(saveDict, self.checkpointDir)
        return self.checkpointDir
