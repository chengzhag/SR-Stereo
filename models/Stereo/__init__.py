import os
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from evaluation import evalFcn
from utils import myUtils
from .PSMNet import stackhourglass as getPSMNet
from .PSMNet_TieCheng import stackhourglass as getPSMNet_TieCheng
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
        self.outputMaxDisp = maxdisp * dispScale # final output value range of disparity map

    def initModel(self):
        self.model = self.getModel(self.maxdisp)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def trainPrepare(self, batch=()):
        batch = super(Stereo, self).trainPrepare(batch)
        if len(batch) != 0:
            batch.allDisps([disp / self.dispScale if disp is not None else None for disp in batch.allDisps()])
        return batch

    def predictPrepare(self, batch=()):
        batch = super(Stereo, self).predictPrepare(batch)
        autoPad = myUtils.AutoPad(batch[0], self.multiple)
        return batch, autoPad

    def predict(self, batch, mask=(1, 1)):
        super(Stereo, self).predict(batch)

    def test(self, batch, type='l1', returnOutputs=False, kitti=False):
        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        dispOuts = self.predict(batch.lastScaleBatch(), [disp is not None for disp in disps])
        for gt, dispOut, side in zip(disps, dispOuts, ('L', 'R')):
            if dispOut is not None:
                if dispOut.dim() == 3:
                    dispOut = dispOut.unsqueeze(1)

                if kitti:
                    mask = gt > 0
                    dispOut = dispOut[mask]
                    gt = gt[mask]
                scores[type + side] = evalFcn.getEvalFcn(type)(gt, dispOut)

                outputs['output' + side] = dispOut / self.outputMaxDisp if returnOutputs else None

        return scores, outputs

    def load(self, checkpointDir):
        checkpointDir = super(Stereo, self).beforeLoad(checkpointDir)
        if checkpointDir is None:
            return

        state_dict = torch.load(checkpointDir)

        # def loadValue(name):
        #     if name in state_dict.keys():
        #         value = state_dict[name]
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
        self.initModel()
        self.model.load_state_dict(state_dict['state_dict'])

        print('Loading complete! Number of model parameters: %d' % self.nParams())

    def save(self, epoch, iteration, trainLoss):
        super(Stereo, self).beforeSave(epoch, iteration)
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
            'maxdisp': self.maxdisp,
            'dispScale': self.dispScale,
            'outputMaxDisp': self.outputMaxDisp
        }, self.checkpointDir)
        return self.checkpointDir


class PSMNet(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(PSMNet, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)
        self.getModel = getPSMNet

    def loss(self, outputs, gts, kitti=False):
        outputs = [output.unsqueeze(1) for output in outputs]
        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        mask = (gts > 0) if kitti else (gts < self.maxdisp)
        mask.detach_()
        loss = 0.5 * F.smooth_l1_loss(outputs[0][mask], gts[mask], reduction='mean') + 0.7 * F.smooth_l1_loss(
            outputs[1][mask], gts[mask], reduction='mean') + F.smooth_l1_loss(outputs[2][mask], gts[mask],
                                                                                 reduction='mean')
        return loss

    def trainOneSide(self, imgL, imgR, gt, returnOutputs=False, kitti=False):
        self.optimizer.zero_grad()

        outputs = self.model(imgL, imgR)

        loss = self.loss(outputs, gt, kitti=kitti)
        loss.backward()
        self.optimizer.step()

        return loss.data.item(), outputs[2] if returnOutputs else None

    def train(self, batch, returnOutputs=False, kitti=False, weights=()):
        myUtils.assertBatchLen(batch, 4)
        imgL, imgR, dispL, dispR = super(PSMNet, self).trainPrepare(batch)

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        for inputL, inputR, gt, process, side in zip((imgL, imgR), (imgR, imgL), (dispL, dispR),
                                               (lambda im: im, myUtils.flipLR), ('L', 'R')):
            loss, dispOut = self.trainOneSide(
                process(inputL), process(inputR), process(gt), returnOutputs, kitti
            ) if gt is not None else (None, None)
            losses['loss' + side] = loss
            outputs['output' + side] = (process(dispOut) * self.dispScale / self.outputMaxDisp).cpu()\
                if dispOut is not None else None

        return losses, outputs

    def predict(self, batch, mask=(1, 1)):
        myUtils.assertBatchLen(batch, 4)
        batch, autoPad = super(PSMNet, self).predictPrepare(batch)
        imgs = batch.lowestResRGBs()

        with torch.no_grad():
            imgL, imgR = autoPad.pad(imgs)
            outputs = []
            for inputL, inputR, process, do in zip((imgL, imgR), (imgR, imgL),
                                                   (lambda im: im, myUtils.flipLR), mask):
                outputs.append(
                    autoPad.unpad(
                        process(
                            self.model(process(inputL),
                                       process(inputR)
                                       )
                        ) * self.dispScale
                    ) if do else None
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

        if returnOutputs:
            with torch.no_grad():
                rOutput = self.down(outputs[2])
        else:
            rOutput = None
        losses = [loss,] + losses
        return [loss.data.item() for loss in losses], rOutput

    def train(self, batch, returnOutputs=False, kitti=False, weights=(1, 0)):
        myUtils.assertBatchLen(batch, 8)
        batch = super(PSMNet, self).trainPrepare(batch)

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        imgL, imgR = batch.highResRGBs()
        for inputL, inputR, gts, process, side in zip((imgL, imgR), (imgR, imgL),
                                               zip(batch.highResDisps(), batch.lowResDisps()),
                                               (lambda im: im, myUtils.flipLR), ('L', 'R')):
            lossN, dispOut = self.trainOneSide(
                process(inputL), process(inputR), process(gts), returnOutputs, kitti, weights=weights
            ) if gts is not None else (None, None)
            for i, loss in enumerate(lossN):
                losses['loss' + side + ('' if i == 0 else str(i))] = loss

            outputs['output' + side] =(process(dispOut) * self.dispScale / self.outputMaxDisp).cpu() \
                if dispOut is not None else None

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
