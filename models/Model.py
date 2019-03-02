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

        self.newFolderName = time.strftime('%y%m%d%H%M%S_', self.startTime) \
                              + self.__class__.__name__ \
                              + saveFolderSuffix
        if dataset is not None: self.newFolderName += ('_%s' % dataset)
        self.newFolder = os.path.join('logs', stage, self.newFolderName)
        self.logFolder = None
        self.checkpointDir = None
        self.checkpointFolder = None

        self.getModel = None
        self.model = None
        self.optimizer = None

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
        def scanCheckpoint(checkpointDirs):
            # if checkpoint is folder
            if os.path.isdir(checkpointDirs):
                filenames = [d for d in os.listdir(checkpointDirs) if os.path.isfile(os.path.join(checkpointDirs, d))]
                filenames.sort()
                latestCheckpointName = None
                latestEpoch = None

                def _getEpoch(name):
                    try:
                        return int(name.split('_')[-3])
                    except ValueError:
                        return None

                for filename in filenames:
                    if any(filename.endswith(extension) for extension in ('.tar', '.pt')):
                        if latestCheckpointName is None:
                            latestCheckpointName = filename
                            latestEpoch = _getEpoch(filename)
                        else:
                            epoch = _getEpoch(filename)
                            if epoch > latestEpoch or epoch is None:
                                latestCheckpointName = filename
                                latestEpoch = epoch
                checkpointDirs = os.path.join(checkpointDirs, latestCheckpointName)

            return checkpointDirs

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
                    raise Exception(f'Error: Specified {len(checkpointDirs)} checkpoints. Only {maxCheckPoints} are needed!')
                # for model composed with multiple models, check if checkpointDirs are together
                modelRoot = None
                checkpointDirs = [scanCheckpoint(dir) for dir in checkpointDirs]
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
                if len(checkpointDirs) == maxCheckPoints:
                    self.checkpointDir = checkpointDirs[-1]
                    self.checkpointFolder, _ = os.path.split(self.checkpointDir)
                else:
                    self.checkpointFolder = self.newFolder
                    # If model is composed with multiple models, save logs to a new folder
                self.logFolder = os.path.join(self.checkpointFolder, 'logs')

        if type(checkpointDirs) is str:
            checkpointDirs = scanCheckpoint(checkpointDirs)

            # update checkpointDir
            self.checkpointDir = checkpointDirs
            self.checkpointFolder, _ = os.path.split(self.checkpointDir)

            # When testing, log files should be saved to checkpointFolder.
            # Here checkpointFolder is setted as default logging folder.
            self.logFolder = os.path.join(self.checkpointFolder, 'logs')

        if self.model is None:
            self.initModel()
        return checkpointDirs

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def savePrepare(self, epoch, iteration, toOld=False):
        # update checkpointDir
        self.checkpointFolder = self.checkpointFolder if toOld else self.newFolder
        self.checkpointDir = os.path.join(self.checkpointFolder,
                                          'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        self.logFolder = os.path.join(self.checkpointFolder, 'logs')
        myUtils.checkDir(self.checkpointFolder)
