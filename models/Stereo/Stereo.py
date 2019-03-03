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

    def predict(self, batch, mask=(1, 1)):
        super(Stereo, self).predict(batch)

    def test(self, batch, type='l1', returnOutputs=False, kitti=False):
        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        mask = [disp is not None for disp in disps]
        dispOuts = self.predict(batch.lastScaleBatch(), mask)
        for gt, dispOut, side in zip(disps, dispOuts, ('L', 'R')):
            if dispOut is not None:
                if dispOut.dim() == 3:
                    dispOut = dispOut.unsqueeze(1)

                # for kitti dataset, only consider loss of none zero disparity pixels in gt
                if kitti:
                    mask = gt > 0
                    dispOut = dispOut[mask]
                    gt = gt[mask]

                scores[type + side] = evalFcn.getEvalFcn(type)(gt, dispOut)

                if returnOutputs:
                    outputs['output' + side] = dispOut / self.outputMaxDisp

        return scores, outputs

    def load(self, checkpointDir):
        checkpointDir = self.loadPrepare(checkpointDir)
        if checkpointDir is None:
            return None, None

        loadStateDict = torch.load(checkpointDir)

        self.model.load_state_dict(loadStateDict['state_dict'])
        if 'optimizer' in loadStateDict.keys():
            self.optimizer.load_state_dict(loadStateDict['optimizer'])
        print('Loading complete! Number of model parameters: %d' % self.nParams())

        epoch = loadStateDict.get('epoch')
        iteration = loadStateDict.get('iteration')
        print(f'Checkpoint epoch {epoch}, iteration {iteration}')
        return epoch, iteration

    def save(self, epoch, iteration, trainLoss):
        self.savePrepare(epoch, iteration)
        saveDict ={
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
            'maxdisp': self.maxdisp,
            'dispScale': self.dispScale,
            'outputMaxDisp': self.outputMaxDisp
        }
        if self.optimizer is not None:
            saveDict['optimizer'] = self.optimizer.state_dict()
        torch.save(saveDict, self.checkpointDir)
        return self.checkpointDir

