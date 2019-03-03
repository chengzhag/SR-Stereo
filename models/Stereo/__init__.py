import os
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from evaluation import evalFcn
from utils import myUtils
from .PSMNet import stackhourglass as rawPSMNet
from .PSMNet_TieCheng import stackhourglass as rawPSMNet_TieCheng
from ..Model import Model
from .. import SR
import collections
import torch.nn.parallel as P


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

    def trainPrepare(self, batch):
        super(Stereo, self).trainPrepare()
        if len(batch) != 0:
            batch.allDisps([disp / self.dispScale if disp is not None else None for disp in batch.allDisps()])
        return batch

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
        checkpointDir = super(Stereo, self).loadPrepare(checkpointDir)
        if checkpointDir is None:
            return None, None

        loadStateDict = torch.load(checkpointDir)

        # def loadValue(name):
        #     if name in state_dict.keys():
        #         value = loadStateDict[name]
        #         if value != getattr(self, name):
        #             print(
        #                 f'Specified {name} \'{getattr(self, name)}\' from args '
        #                 f'is not equal to {name} \'{value}\' loaded from checkpoint!'
        #                 f' Using loaded {name} instead!')
        #         setattr(self, name, value)
        #     else:
        #         print(f'No {name} found in checkpoint! Using {name} \'{getattr(self, name)}\' specified in args!')

        # loadValue('maxdisp')
        # loadValue('dispScale')
        self.model.load_state_dict(loadStateDict['state_dict'])
        if 'optimizer' in loadStateDict.keys():
            self.optimizer.load_state_dict(loadStateDict['optimizer'])
        print('Loading complete! Number of model parameters: %d' % self.nParams())

        epoch = loadStateDict.get('epoch')
        iteration = loadStateDict.get('iteration')
        print(f'Checkpoint epoch {epoch}, iteration {iteration}')
        return epoch, iteration

    def save(self, epoch, iteration, trainLoss, toOld=False):
        super(Stereo, self).savePrepare(epoch, iteration, toOld)
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

class rawPSMNetScale(rawPSMNet):
    def __init__(self, maxdisp, dispScale=1, multiple=16):
        super(rawPSMNetScale, self).__init__(maxdisp)
        self.multiple = multiple
        self.__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225]}
        self.dispScale = dispScale

    def forward(self, left, right):
        def normalize(nTensor):
            nTensorClone = nTensor.clone()
            for tensor in nTensorClone:
                for t, m, s in zip(tensor, self.__imagenet_stats['mean'], self.__imagenet_stats['std']):
                    t.sub_(m).div_(s)
            return nTensorClone

        left, right = normalize(left), normalize(right)

        if self.training:
            outputs = super(rawPSMNetScale, self).forward(left, right)
        else:
            autoPad = myUtils.AutoPad(left, self.multiple)

            left, right = autoPad.pad((left, right))
            outputs = super(rawPSMNetScale, self).forward(left, right)
            outputs = autoPad.unpad(outputs)
        # outputs = myUtils.forNestingList(outputs, lambda disp: disp * self.dispScale)
        return outputs

class PSMNet(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(PSMNet, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)

        self.getModel = rawPSMNetScale

    def initModel(self):
        self.model = self.getModel(self.maxdisp)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def loss(self, outputs, gts, kitti=False):
        outputs = [output.unsqueeze(1) for output in outputs]
        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        mask = (gts.detach() > 0) if kitti else (gts.detach() < self.maxdisp)
        loss = 0.5 * F.smooth_l1_loss(outputs[0][mask], gts[mask], reduction='mean') + 0.7 * F.smooth_l1_loss(
            outputs[1][mask], gts[mask], reduction='mean') + F.smooth_l1_loss(outputs[2][mask], gts[mask],
                                                                              reduction='mean')
        return loss

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale
    def forward(self, imgL, imgR):
        outputs = self.model(imgL, imgR)
        return outputs

    def trainOneSide(self, imgL, imgR, gt, returnOutputs=False, kitti=False):
        self.optimizer.zero_grad()
        outputs = self.forward(imgL, imgR)
        loss = self.loss(outputs, gt, kitti=kitti)
        loss.backward()
        self.optimizer.step()

        return loss.data.item(), outputs[2].detach() * self.dispScale / self.outputMaxDisp if returnOutputs else None

    def trainPrepare(self, batch):
        batch = super(PSMNet, self).trainPrepare(batch)
        return batch

    def train(self, batch, returnOutputs=False, kitti=False, weights=()):
        myUtils.assertBatchLen(batch, 4)
        batch = self.trainPrepare(batch)
        imgL, imgR = batch.highResRGBs()

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        for inputL, inputR, gt, process, side in zip(
                (imgL, imgR), (imgR, imgL), batch.highResDisps(),
                (lambda im: im, myUtils.flipLR), ('L', 'R')
        ):
            if gt is not None:
                loss, dispOut = self.trainOneSide(
                    *process([inputL, inputR, gt]), returnOutputs=returnOutputs, kitti=kitti
                )
                losses['loss' + side] = loss
                if returnOutputs:
                    outputs['output' + side] = process(dispOut)

        return losses, outputs

    def predict(self, batch, mask=(1, 1)):
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()
        imgL, imgR = batch.lowestResRGBs()

        with torch.no_grad():
            outputs = []
            for inputL, inputR, process, do in zip((imgL, imgR), (imgR, imgL),
                                                   (lambda im: im, myUtils.flipLR), mask):
                outputs.append(
                    process(
                        self.model(process(inputL),
                                   process(inputR)
                                   )
                    ) * self.dispScale if do else None
                )

            return tuple(outputs)


class PSMNetDown(PSMNet):

    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(PSMNetDown, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)
        self.outputMaxDisp = self.outputMaxDisp // 2

        # Downsampling net
        class AvgDownSample(torch.nn.Module):
            def __init__(self):
                super(AvgDownSample, self).__init__()
                self.pool = nn.AvgPool2d((2, 2))

            def forward(self, x):
                return self.pool(x) / 2

        self.down = AvgDownSample()

    def initModel(self):
        super(PSMNetDown, self).initModel()
        if self.cuda:
            self.down = nn.DataParallel(self.down)
            self.down.cuda()

    def loss(self, outputs, gts, kitti=False):
        losses = []
        losses.append(super(PSMNetDown, self).loss(outputs, gts[0], kitti=kitti))
        outputs = [self.down(output) for output in outputs]
        losses.append(super(PSMNetDown, self).loss(outputs, gts[1], kitti=kitti))
        return losses

    def trainOneSide(self, imgL, imgR, gts, returnOutputs=False, kitti=False, weights=(1, 0)):
        self.optimizer.zero_grad()

        outputs = self.model(imgL, imgR)

        losses = self.loss(outputs, gts, kitti=kitti)
        loss = sum([weight * loss for weight, loss in zip(weights, losses)])
        loss.backward()
        self.optimizer.step()

        dispOuts = []
        if returnOutputs:
            dispOuts.append(outputs[2].detach() * self.dispScale / self.outputMaxDisp / 2)
            with torch.no_grad():
                dispOuts.append(self.down(outputs[2].detach()) * self.dispScale / self.outputMaxDisp)
        losses = [loss, ] + losses
        return [loss.data.item() for loss in losses], dispOuts

    def train(self, batch, returnOutputs=False, kitti=False, weights=(1, 0)):
        myUtils.assertBatchLen(batch, 8)
        batch = self.trainPrepare(batch)

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        imgL, imgR = batch.highResRGBs()
        for inputL, inputR, gts, process, side in zip((imgL, imgR), (imgR, imgL),
                                                      zip(batch.highResDisps(), batch.lowResDisps()),
                                                      (lambda im: im, myUtils.flipLR), ('L', 'R')):
            lossN, dispOuts = self.trainOneSide(
                process(inputL), process(inputR), process(gts), returnOutputs, kitti, weights=weights
            ) if gts is not None else (None, None)
            for suffix, loss in zip(('', 'High', 'Low'), lossN):
                losses['loss' + suffix + side] = loss

            if returnOutputs:
                outputs['outputHigh' + side], outputs['outputLow' + side] = process(dispOuts)

        return losses, outputs

    def predict(self, batch, mask=(1, 1)):
        myUtils.assertBatchLen(batch, 4)
        outputs = super(PSMNetDown, self).predict(batch, mask)
        downsampled = []
        for output in outputs:
            # Down sample to half size
            downsampled.append(self.down(output) if output is not None else None)
        return downsampled

    def test(self, batch, type='l1', returnOutputs=False, kitti=False):
        myUtils.assertBatchLen(batch, 8)
        batch = myUtils.Batch(batch.highResRGBs() + batch.lowestResDisps())
        return super(PSMNetDown, self).test(batch, type, returnOutputs, kitti)


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
        super(SRStereo, self).beforeLoad(checkpointDir)
        if type(checkpointDir) not in (list, tuple) or len(checkpointDir) != 2:
            raise Exception('Error: SRStereo need two checkpoints (SR/Stereo) to load!')
        self.sr.load(checkpointDir[0])
        self.stereo.load(checkpointDir[1])


class PSMNet_TieCheng(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(PSMNet_TieCheng, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)
        self.getModel = getPSMNet_TieCheng

    def predict(self, batch, mask=(1, 1)):
        batch, autoPad = super(PSMNet_TieCheng, self).predictPrepare(batch)
        inputs = batch.lowestResRGBs()

        with torch.no_grad():
            imgL, imgR = autoPad.pad(inputs)
            outputs = self.model(imgL, imgR)
            outputs = autoPad.unpad(outputs)
            return tuple(outputs)
