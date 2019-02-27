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


class Stereo(Model):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(Stereo, self).__init__(cuda, half, stage, dataset, saveFolderSuffix)
        self.maxdisp = maxdisp
        self.dispScale = dispScale

    def initModel(self):
        self.model = self.getModel(round(self.maxdisp // self.dispScale))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def train(self, imgL, imgR, dispL=None, dispR=None):
        super(Stereo, self).trainPrepare()
        myUtils.assertDisp(dispL, dispR)
        if self.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            dispL = dispL.cuda() if dispL is not None else None
            dispR = dispR.cuda() if dispR is not None else None
        return imgL, imgR, dispL, dispR

    def predict(self, imgL, imgR, mask=(1, 1)):
        super(Stereo, self).predictPrepare()
        autoPad = myUtils.AutoPad(imgL, self.multiple)
        return autoPad

    def test(self, imgL, imgR, dispL=None, dispR=None, type='l1', returnOutputs=False, kitti=False):
        myUtils.assertDisp(dispL, dispR)

        if self.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            dispL = dispL.cuda() if dispL is not None else None
            dispR = dispR.cuda() if dispR is not None else None

        # for kitti dataset, only consider loss of none zero disparity pixels in gt

        scores = []
        outputs = []
        dispOuts = self.predict(imgL, imgR, [disp is not None for disp in (dispL, dispR)])
        for gt, dispOut in zip([dispL, dispR], dispOuts):
            if dispOut is not None:
                if dispOut.dim() == 3:
                    dispOut = dispOut.unsqueeze(1)
                outputs.append(dispOut if returnOutputs else None)

                if kitti:
                    mask = gt > 0
                    dispOut = dispOut[mask]
                    gt = gt[mask]
                scores.append(evalFcn.getEvalFcn(type)(gt, dispOut))
            else:
                outputs.append(None)
                scores.append(None)

        return scores, outputs

    def load(self, checkpointDir):
        super(Stereo, self).load(checkpointDir)

        state_dict = torch.load(checkpointDir)

        def loadValue(name):
            if name in state_dict.keys():
                value = state_dict[name]
                if value != getattr(self, name):
                    print(
                        f'Specified {name} \'{getattr(self, name)}\' from args '
                        f'is not equal to {name} \'{value}\' loaded from checkpoint!'
                        f' Using loaded {name} instead!')
                setattr(self, name, value)
            else:
                print(f'No {name} found in checkpoint! Using {name} \'{getattr(self, name)}\' specified in args!')

        loadValue('maxdisp')
        loadValue('dispScale')
        self.initModel()
        self.model.load_state_dict(state_dict['state_dict'])

        print('Loading complete! Number of model parameters: %d' % self.nParams())

    def save(self, epoch, iteration, trainLoss):
        super(Stereo, self)._save(epoch, iteration)
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
            'maxdisp': self.maxdisp,
            'dispScale': self.dispScale
        }, self.checkpointDir)
        return self.checkpointDir


class PSMNet(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(PSMNet, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)
        self.getModel = getPSMNet

    def _train_original(self, imgL, imgR, disp_true, output=False, kitti=False):
        self.optimizer.zero_grad()

        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        mask = (disp_true > 0) if kitti else (disp_true < self.maxdisp)
        mask.detach_()

        output1, output2, output3 = self.model(imgL, imgR)
        output1 = output1.unsqueeze(1)
        output2 = output2.unsqueeze(1)
        output3 = output3.unsqueeze(1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean') + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], reduction='mean') + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                 reduction='mean')

        loss.backward()
        self.optimizer.step()

        return loss.data.item(), output3 if output else None

    def train(self, imgL, imgR, dispL=None, dispR=None, output=False, kitti=False):
        imgL, imgR, dispL, dispR = super(PSMNet, self).train(imgL, imgR, dispL, dispR)
        dispL, dispR = dispL / self.dispScale if dispL is not None else None, \
                       dispR / self.dispScale if dispR is not None else None

        losses = []
        outputs = []
        for inputL, inputR, gt, process in zip((imgL, imgR), (imgR, imgL), (dispL, dispR),
                                               (lambda im: im, myUtils.flipLR)):
            loss, dispOut = self._train_original(
                process(inputL), process(inputR), process(gt), output, kitti
            ) if gt is not None else (None, None)
            losses.append(loss)
            outputs.append(
                (process(dispOut) * self.dispScale).cpu()
                if dispOut is not None else None
            )

        return losses, outputs

    def predict(self, imgL, imgR, mask=(1, 1)):
        autoPad = super(PSMNet, self).predict(imgL, imgR, mask)

        with torch.no_grad():
            imgL, imgR = autoPad.pad(imgL, self.cuda), autoPad.pad(imgR, self.cuda)
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

    def train(self, imgL, imgR, dispL=None, dispR=None, output=False, kitti=False):
        raise Exception('Error: fcn train() not completed yet!')

    def predict(self, imgL, imgR, mask=(1, 1)):
        outputs = super(PSMNetDown, self).predict(imgL, imgR, mask)
        downsampled = []
        for output in outputs:
            # Down sample to half size
            downsampled.append(self.down(output) if output is not None else None)
        return downsampled


class PSMNet_TieCheng(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
                 saveFolderSuffix=''):
        super(PSMNet_TieCheng, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)
        self.getModel = getPSMNet_TieCheng

    def train(self, imgL, imgR, dispL=None, dispR=None, output=False, kitti=False):
        imgL, imgR, dispL, dispR = super(PSMNet_TieCheng, self).train(imgL, imgR, dispL, dispR)
        raise Exception('Fcn \'train\' not done yet...')

    def predict(self, imgL, imgR, mode='both'):
        autoPad = super(PSMNet_TieCheng, self).predict(imgL, imgR, mode)

        with torch.no_grad():
            imgL, imgR = autoPad.pad(imgL, self.cuda), autoPad.pad(imgR, self.cuda)
            pl, pr = self.model(imgL, imgR)
            if mode == 'left':
                return autoPad.unpad(pl) * self.dispScale
            elif mode == 'right':
                return autoPad.unpad(pr) * self.dispScale
            elif mode == 'both':
                return autoPad.unpad(pl) * self.dispScale, autoPad.unpad(pr) * self.dispScale
            else:
                raise Exception('No mode \'%s\'!' % mode)
