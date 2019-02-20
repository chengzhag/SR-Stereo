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


class Stereo:
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, stage='unnamed', dataset=None, saveFolderSuffix=''):
        self.maxdisp = maxdisp
        self.dispScale = dispScale
        self.cuda = cuda
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
        self.model = self.getModel(round(self.maxdisp // self.dispScale))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def train(self, imgL, imgR, dispL=None, dispR=None):
        if self.model is None:
            self.initModel()
        # When training, log files should be saved to saveFolder.
        self.logFolder = os.path.join(self.saveFolder, 'logs')
        self.model.train()
        myUtils.assertDisp(dispL, dispR)
        if self.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            dispL = dispL.cuda() if dispL is not None else None
            dispR = dispR.cuda() if dispR is not None else None
        return imgL, imgR, dispL, dispR

    def predict(self, imgL, imgR, mode='both'):
        self.model.eval()
        autoPad = myUtils.AutoPad(imgL, self.multiple)
        return autoPad

    def test(self, imgL, imgR, dispL=None, dispR=None, type='l1', output=False, kitti=False):
        myUtils.assertDisp(dispL, dispR)

        if self.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            dispL = dispL.cuda() if dispL is not None else None
            dispR = dispR.cuda() if dispR is not None else None

        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        scores = []
        outputs = []
        for gt, mode in zip([dispL, dispR], ['left', 'right']):
            if gt is None:
                scores.append(None)
                outputs.append(None)
                continue
            dispOut = self.predict(imgL, imgR, mode)
            if output:
                outputs.append(dispOut.cpu())
            if kitti:
                mask = gt > 0
                dispOut = dispOut[mask]
                gt = gt[mask]
            scores.append(getattr(evalFcn, type)(gt, dispOut))

        if output:
            return scores, outputs
        else:
            return scores

    def load(self, checkpointDir):
        if checkpointDir is not None:
            print('Loading checkpoint from %s' % checkpointDir)
            state_dict = torch.load(checkpointDir)

            def loadValue(name):
                if name in state_dict.keys():
                    value = state_dict[name]
                    if value != getattr(self, name):
                        print(
                            f'Specified {name} \'{getattr(self, name)}\' from args is not equal to {name} \'{value}\' loaded from checkpoint! Using loaded {name} instead!')
                    setattr(self, name, value)
                else:
                    print(f'No {name} find in checkpoint! Using {name} \'{getattr(self, name)}\' specified in args!')

            loadValue('maxdisp')
            loadValue('dispScale')
            self.initModel()
            self.model.load_state_dict(state_dict['state_dict'])

            # update checkpointDir
            self.checkpointDir = checkpointDir
            self.checkpointFolder, _ = os.path.split(self.checkpointDir)
            # When testing, log files should be saved to checkpointFolder.
            # Here checkpointFolder is setted as default logging folder.
            self.logFolder = os.path.join(self.checkpointFolder, 'logs')

            print('Loading complete! Number of model parameters: %d' % self.nParams())
        else:
            raise Exception('checkpoint dir is None!')

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def save(self, epoch, iteration, trainLoss):
        # update checkpointDir
        self.checkpointDir = os.path.join(self.saveFolder, 'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        self.checkpointFolder = self.saveFolder
        self.logFolder = os.path.join(self.checkpointFolder, 'logs')

        myUtils.checkDir(self.saveFolder)
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
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(PSMNet, self).__init__(maxdisp, dispScale, cuda, stage, dataset, saveFolderSuffix)
        self.getModel = getPSMNet

    def train(self, imgL, imgR, dispL=None, dispR=None, output=True, kitti=False):
        imgL, imgR, dispL, dispR = super(PSMNet, self).train(imgL, imgR, dispL, dispR)
        dispL, dispR = dispL / self.dispScale, dispR / self.dispScale

        def _train(imgL, imgR, disp_true):
            self.optimizer.zero_grad()

            # for kitti dataset, only consider loss of none zero disparity pixels in gt
            mask = (disp_true > 0) if kitti else (disp_true < self.maxdisp)
            mask.detach_()

            output1, output2, output3 = self.model(imgL, imgR)
            output1 = torch.squeeze(output1, 1)
            output2 = torch.squeeze(output2, 1)
            output3 = torch.squeeze(output3, 1)
            loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean') + 0.7 * F.smooth_l1_loss(
                output2[mask], disp_true[mask], reduction='mean') + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                     reduction='mean')

            loss.backward()
            self.optimizer.step()
            if output:
                return loss.data.item(), output3
            else:
                return loss.data.item()

        losses = []
        if output:
            outputs = []
            if dispL is not None:
                loss, dispOut = _train(imgL, imgR, dispL)
                losses.append(loss)
                outputs.append((dispOut * self.dispScale).cpu())
            else:
                losses.append(None)
                outputs.append(None)

            if dispR is not None:
                # swap and flip input for right disparity map
                loss, dispOut = _train(myUtils.flipLR(imgR), myUtils.flipLR(imgL),
                                       myUtils.flipLR(dispR))
                losses.append(loss)
                outputs.append((myUtils.flipLR(dispOut) * self.dispScale).cpu())
            else:
                losses.append(None)
                outputs.append(None)

            return losses, outputs
        else:
            losses.append(_train(imgL, imgR, dispL) if dispL is not None else None)
            # swap and flip input for right disparity map
            losses.append(_train(myUtils.flipLR(imgR), myUtils.flipLR(imgL),
                                 myUtils.flipLR(dispR)) if dispR is not None else None)

            return losses

    def predict(self, imgL, imgR, mode='both'):
        autoPad = super(PSMNet, self).predict(imgL, imgR, mode='both')

        with torch.no_grad():
            def predictL():
                return autoPad.unpad(self.model(imgL, imgR)) * self.dispScale

            def predictR():
                return autoPad.unpad(
                    myUtils.flipLR(self.model(myUtils.flipLR(imgR), myUtils.flipLR(imgL)))
                ) * self.dispScale

            imgL, imgR = autoPad.pad(imgL, self.cuda), autoPad.pad(imgR, self.cuda)
            if mode == 'left':
                return predictL()
            elif mode == 'right':
                return predictR()
            elif mode == 'both':
                return predictL(), predictR()
            else:
                raise Exception('No mode \'%s\'!' % mode)


class PSMNet_TieCheng(Stereo):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, maxdisp=192, dispScale=1, cuda=True, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(PSMNet_TieCheng, self).__init__(maxdisp, dispScale, cuda, stage, dataset, saveFolderSuffix)
        self.getModel = getPSMNet_TieCheng

    def train(self, imgL, imgR, dispL=None, dispR=None, output=True, kitti=False):
        imgL, imgR, dispL, dispR = super(PSMNet_TieCheng, self).train(imgL, imgR, dispL, dispR)
        raise Exception('Fcn \'train\' not done yet...')

    def predict(self, imgL, imgR, mode='both'):
        autoPad = super(PSMNet_TieCheng, self).predict(imgL, imgR, mode='both')

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
