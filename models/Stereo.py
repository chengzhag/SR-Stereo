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

    def train(self, imgL, imgR, disp_L, disp_R=None):

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

        self.model.train()
        loss = _train(self, imgL, imgR, disp_L)

        if disp_R is not None:
            # swap and flip input for right disparity map
            loss = loss + _train(self, imgR.flip(3), imgL.flip(3), disp_R.flip(2))
            loss = loss / 2

        return loss

    def test(self, imgL, imgR, disp_true):
        self.model.eval()
        # imgL = Variable(torch.FloatTensor(imgL))
        # imgR = Variable(torch.FloatTensor(imgR))

        mask = disp_true < 192

        with torch.no_grad():
            output3 = self.model(imgL, imgR)

        output = torch.squeeze(output3.data.cpu(), 1)[:, 4:, :]

        if len(disp_true[mask]) == 0:
            loss = 0
        else:
            loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error

        return loss



