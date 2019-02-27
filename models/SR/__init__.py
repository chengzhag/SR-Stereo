import os
import time
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from evaluation import evalFcn
from utils import myUtils
from .EDSR import edsr
from ..Model import Model
import torch.nn.parallel as P
from models.SR.warp import warp


class SR(Model):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, cInput=3, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SR, self).__init__(cuda, half, stage, dataset, saveFolderSuffix)

        class Arg:
            def __init__(self):
                self.n_resblocks = 16
                self.n_feats = 64
                self.scale = [2]
                self.rgb_range = 255
                self.n_colors = 3
                self.n_inputs = cInput
                self.res_scale = 1

        self.args = Arg()

    def initModel(self):
        self.model = edsr.make_model(self.args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model.cuda()

    # imgL: RGB value range 0~1
    # imgH: RGB value range 0~1
    def train(self, imgL, imgH):
        super(SR, self).trainPrepare()

        if self.cuda:
            imgL, imgH = imgL.cuda(), imgH.cuda()
        self.optimizer.zero_grad()
        output = P.data_parallel(self.model, imgL * self.args.rgb_range)
        loss = F.smooth_l1_loss(imgH * self.args.rgb_range, output, reduction='mean')
        with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
        output = output / self.args.rgb_range
        return loss.data.item(), output

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, imgL):
        super(SR, self).predictPrepare()
        with torch.no_grad():
            output = P.data_parallel(self.model, imgL * self.args.rgb_range)
            output = myUtils.quantize(output, self.args.rgb_range) / self.args.rgb_range
            return output

    def test(self, imgL, imgH, type='l1'):
        if self.cuda:
            imgL, imgH = imgL.cuda(), imgH.cuda()

        output = self.predict(imgL)
        score = evalFcn.getEvalFcn(type)(imgH * self.args.rgb_range, output * self.args.rgb_range)

        return score, output

    def load(self, checkpointDir):
        super(SR, self).load(checkpointDir)

        self.initModel()
        load_state_dict = torch.load(checkpointDir)
        if 'state_dict' in load_state_dict.keys():
            load_state_dict = load_state_dict['state_dict']

        model_dict = self.model.state_dict()
        selected_load_dict = {}
        for loadName, loadValue in load_state_dict.items():
            if loadName in model_dict and model_dict[loadName].size() == loadValue.size():
                selected_load_dict[loadName] = loadValue
            else:
                # try to initialize input layer from weights with different channels
                # if loadName == 'head.0.weight':
                #     selected_load_dict[loadName] = model_dict[loadName]
                #     selected_load_dict[loadName][:,:3,:,:] = loadValue
                #     selected_load_dict[loadName][:, 3:6, :, :] = loadValue
                print('Warning! While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(loadName, model_dict[loadName].size(), loadValue.size()))
        model_dict.update(selected_load_dict)

        self.model.load_state_dict(model_dict, strict=False)

        print('Loading complete! Number of model parameters: %d' % self.nParams())

    def save(self, epoch, iteration, trainLoss):
        super(SR, self)._save(epoch, iteration)

        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
        }, self.checkpointDir)

class SRdisp(SR):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, withMask=False, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SRdisp, self).__init__(7 if withMask else 6, cuda, half, stage, dataset, saveFolderSuffix)
        self.withMask = withMask

    def initModel(self):
        super(SRdisp, self).initModel()


    def _preProcess(self, inputL, inputR, dispL, dispR):
        with torch.no_grad():
            warpToL, warpToR, maskL, maskR = warp(inputL, inputR, dispL, dispR)

            inputs = []
            for input in zip((inputL, inputR), (warpToL, warpToR), (maskL, maskR)):
                if self.args.n_inputs == 7:
                    inputs.append(torch.cat(input, 1))
                elif self.args.n_inputs == 6:
                    inputs.append(torch.cat(input[:2], 1))
                else:
                    raise Exception(
                        'Error: self.model.args.n_inputs = %d which is not supporty!' % self.model.args.n_inputs)
            return inputs
    # imgL: RGB value range 0~1
    # imgH: RGB value range 0~1
    def train(self, batch, output=False):
        inputs = self._preProcess(*batch[4:8])

        losses = []
        outputs = []
        for input, gt in zip(inputs, batch[0:2]):
            loss, predict = super(SRdisp, self).train(input, gt) if gt is not None else (None, None)
            losses.append(loss)
            outputs.append(myUtils.quantize(predict, 1) if output and predict is not None else None)

        return losses, outputs

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
        inputs = self._preProcess(*batch)
        outputs = []
        for input, do in zip(inputs, mask):
            outputs.append(super(SRdisp, self).predict(input) if do else None)

        return tuple(outputs)

    def test(self, batch, type='l1', output=False):
        scores = []
        outputs = []
        predicts = self.predict(batch[4:8], mask=[gt is not None for gt in batch[0:2]])
        for gt, predict in zip(batch[0:2], predicts):
            if predict is not None:
                scores.append(evalFcn.getEvalFcn(type)(gt * self.args.rgb_range, predict * self.args.rgb_range))
                outputs.append(predict if output else None)
            else:
                outputs.append(None)
                scores.append(None)

        return scores, outputs

