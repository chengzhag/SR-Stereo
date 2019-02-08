from models.PSMNet import *
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def Stereo(maxdisp=192, model='PSMNet'):
    if model == 'PSMNet':
        return _Stereo_PSMNet(stackhourglass(maxdisp))
    else:
        print('no model')

class _Stereo_PSMNet():
    def __init__(self, PSMNet):
        self.model = PSMNet
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.maxdisp = PSMNet.maxdisp

    def train(self, imgL, imgR, dispL=None, dispR=None):
        self.model.train()
        self._assertDisp(dispL, dispR)
        def _train(self, imgL, imgR, disp_true):
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
            losses.append(_train(self, imgL, imgR, dispL))

        if dispR is not None:
            # swap and flip input for right disparity map
            losses.append(_train(self, self._flip(imgR), self._flip(imgL), self._flip(dispR)))

        loss = sum(losses)/len(losses)

        return loss, losses

    def predict(self, imgL, imgR, mode='both'):
        self.model.eval()
        def _predictL(self, imgL, imgR):return self.model(imgL, imgR)
        def _predictR(self, imgL, imgR):return self._flip(self.model(self._flip(imgR), self._flip(imgL)))

        with torch.no_grad():
            if mode == 'left':
                return _predictL(self, imgL, imgR)
            elif mode == 'right':
                return _predictR(self, imgL, imgR)
            elif mode == 'both':
                return _predictL(self, imgL, imgR), _predictR(self, imgL, imgR)
            else:
                raise Exception('No mode \'%s\'!' % mode)

    def test(self, imgL, imgR, dispL=None, dispR=None, type='l1'):
        self._assertDisp(dispL, dispR)

        def _test(fcn, imgL, imgR, dispL=None, dispR=None):
            losses = []

            for gt, mode in zip([dispL, dispR], ['left', 'right']):
                if gt is None:
                    continue
                output = self.predict(imgL, imgR, mode)
                mask = gt < self.maxdisp
                output = torch.squeeze(output.data.cpu(), 1)[:, 4:, :]
                losses.append(fcn(gt[mask], output[mask]).data)

            loss = sum(losses) / len(losses)
            return loss, losses

        if type == 'l1':
            def l1Loss(gt, output):
                if len(gt) == 0:
                    loss = 0
                else:
                    loss = torch.mean(torch.abs(output - gt))  # end-point-error
                return loss
            return _test(l1Loss, imgL, imgR, dispL, dispR)

        else:
            raise Exception('No error type \'%s\'!' % type)


    def _flip(self, im):
        return im.flip(-1)

    def _assertDisp(self, dispL=None, dispR=None):
        if dispL is None and dispR is None:
            raise Exception('No disp input!')



