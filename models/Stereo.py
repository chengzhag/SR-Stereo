from models.PSMNet import *
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from evaluation import evalFcn


def Stereo(maxdisp=192, model='PSMNet', cuda=True):
    if model == 'PSMNet':
        return _Stereo_PSMNet(stackhourglass(maxdisp), cuda=cuda)
    else:
        print('no model')


class _Stereo_PSMNet():
    def __init__(self, PSMNet, cuda = True):
        self.model = PSMNet
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.maxdisp = PSMNet.maxdisp
        self._cuda()

    def train(self, imgL, imgR, dispL=None, dispR=None):
        self.model.train()
        self._assertDisp(dispL, dispR)

        def _train(imgL, imgR, disp_true):
            self.optimizer.zero_grad()

            mask = disp_true < self.maxdisp
            mask.detach_()

            output1, output2, output3 = self.model(imgL, imgR)
            output1 = torch.squeeze(output1, 1)
            output2 = torch.squeeze(output2, 1)
            output3 = torch.squeeze(output3, 1)
            loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
                output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                      size_average=True)

            loss.backward()
            self.optimizer.step()
            return loss.data

        losses = []
        if dispL is not None:
            losses.append(_train(imgL, imgR, dispL))

        if dispR is not None:
            # swap and flip input for right disparity map
            losses.append(_train(self._flip(imgR), self._flip(imgL), self._flip(dispR)))

        loss = sum(losses) / len(losses)

        return loss, losses

    def predict(self, imgL, imgR, mode='both'):
        self.model.eval()

        def _predictL():
            return self.model(imgL, imgR)

        def _predictR():
            return self._flip(self.model(self._flip(imgR), self._flip(imgL)))

        with torch.no_grad():
            if mode == 'left':
                return _predictL()
            elif mode == 'right':
                return _predictR()
            elif mode == 'both':
                return _predictL(), _predictR()
            else:
                raise Exception('No mode \'%s\'!' % mode)

    def test(self, imgL, imgR, dispL=None, dispR=None, type='l1', kitti=False):
        self._assertDisp(dispL, dispR)

        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        scores = []
        for gt, mode in zip([dispL, dispR], ['left', 'right']):
            if gt is None:
                continue
            output = self.predict(imgL, imgR, mode)
            mask = (gt < self.maxdisp) & (gt > 0) if kitti else (gt < self.maxdisp)
            output = torch.squeeze(output.data.cpu(), 1)[:, 4:, :]  # TODO: generalize padding and unpadding process
            scores.append(getattr(evalFcn, type)(gt[mask], output[mask]).data)

        scoreAvg = sum(scores) / len(scores)
        return scoreAvg, scores

    def _flip(self, im):
        return im.flip(-1)

    def _assertDisp(self, dispL=None, dispR=None):
        if dispL is None and dispR is None:
            raise Exception('No disp input!')

    def _cuda(self):
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
