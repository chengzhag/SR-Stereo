import os
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.PSMNet import *
from evaluation import evalFcn


class PSMNet:
    def __init__(self, maxdisp=192, cuda=True, stage='unnamed'):
        self.stage = stage
        self.startTime = time.localtime(time.time())
        self.saveFolderName = time.strftime('%y%m%d%H%M%S_', self.startTime) + self.__class__.__name__
        self.saveFolder = os.path.join('logs', stage, self.saveFolderName)
        self.logFolder = os.path.join(self.saveFolder, 'logs')
        self.model = stackhourglass(maxdisp)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.maxdisp = maxdisp
        self.cuda = cuda
        self.multiple = 16
        if cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        self.checkpoint = None

    def train(self, imgL, imgR, dispL=None, dispR=None, output=True):
        self.model.train()
        self._assertDisp(dispL, dispR)
        if self.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            dispL = dispL.cuda() if dispL is not None else None
            dispR = dispR.cuda() if dispR is not None else None

        def _train(imgL, imgR, disp_true):
            self.optimizer.zero_grad()

            mask = disp_true < self.maxdisp
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
            ouputs = []
            if dispL is not None:
                loss, ouput = _train(imgL, imgR, dispL)
                losses.append(loss)
                ouputs.append(ouput)

            if dispR is not None:
                # swap and flip input for right disparity map
                loss, ouput = _train(self._flip(imgR), self._flip(imgL), self._flip(dispR))
                losses.append(loss)
                ouputs.append(self._flip(ouput))
            loss = sum(losses) / len(losses)
            return loss, losses, ouputs
        else:
            if dispL is not None:
                losses.append(_train(imgL, imgR, dispL))

            if dispR is not None:
                # swap and flip input for right disparity map
                losses.append(_train(self._flip(imgR), self._flip(imgL), self._flip(dispR)))
            loss = sum(losses) / len(losses)
            return loss, losses

    def predict(self, imgL, imgR, mode='both'):
        self.model.eval()

        N, C, H, W = imgL.size()
        HPad = ((H - 1) // self.multiple + 1) * self.multiple
        WPad = ((W - 1) // self.multiple + 1) * self.multiple

        with torch.no_grad():
            def pad(input):
                inputPad = torch.zeros([N, C, HPad, WPad], dtype=input.dtype, device = 'cuda' if self.cuda else 'cpu')
                inputPad[:, :, (HPad - H):, (WPad - W):] = input
                return inputPad

            def unpad(output):
                output = output[:, (HPad - H):, (WPad - W):]
                return output

            def predictL():
                return unpad(self.model(imgL, imgR))

            def predictR():
                return unpad(self._flip(self.model(self._flip(imgR), self._flip(imgL))))

            imgL, imgR = pad(imgL), pad(imgR)
            if mode == 'left':
                return predictL()
            elif mode == 'right':
                return predictR()
            elif mode == 'both':
                return predictL(), predictR()
            else:
                raise Exception('No mode \'%s\'!' % mode)

    def test(self, imgL, imgR, dispL=None, dispR=None, type='l1', kitti=False):
        self._assertDisp(dispL, dispR)
        if self.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            dispL = dispL.cuda() if dispL is not None else None
            dispR = dispR.cuda() if dispR is not None else None
            
        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        scores = []
        for gt, mode in zip([dispL, dispR], ['left', 'right']):
            if gt is None:
                continue
            output = self.predict(imgL, imgR, mode)
            mask = (gt < self.maxdisp) & (gt > 0) if kitti else (gt < self.maxdisp)
            scores.append(getattr(evalFcn, type)(gt[mask], output[mask]))

        scoreAvg = sum(scores) / len(scores)
        return scoreAvg, scores

    def _flip(self, im):
        return im.flip(-1)

    def _assertDisp(self, dispL=None, dispR=None):
        if dispL is None and dispR is None:
            raise Exception('No disp input!')

    def load(self, checkpoint):
        if checkpoint is not None:
            print('Loading checkpoint...')
            state_dict = torch.load(checkpoint)
            self.model.load_state_dict(state_dict['state_dict'])
            self.checkpoint = checkpoint
            print('Loading complete! Number of model parameters: %d' % self.nParams())
        else:
            raise Exception('checkpoint dir is None!')

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def save(self, epoch, iteration, trainLoss):
        self.checkpoint = os.path.join(self.saveFolder, 'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
        }, self.checkpoint)
        return self.checkpoint
