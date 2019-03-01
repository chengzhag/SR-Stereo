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
import collections


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
    def train(self, batch, returnOutputs=False):
        batch = self.trainPrepare(batch)

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        for input, gt, side in zip(batch.lowResRGBs(), batch.highResRGBs(), ('L', 'R')):
            loss, predict = self.trainOneSide(input, gt) if gt is not None else (None, None)
            losses['loss' + side] = loss
            if returnOutputs:
                outputs['output' + side] = myUtils.quantize(predict, 1)

        return losses, outputs

    def trainOneSide(self, imgL, imgH):
        self.optimizer.zero_grad()
        output = P.data_parallel(self.model, imgL * self.args.rgb_range)
        loss = F.smooth_l1_loss(imgH * self.args.rgb_range, output, reduction='mean')
        with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
        output = output.detach() / self.args.rgb_range
        return loss.data.item(), output

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
        batch = self.predictPrepare(batch)
        myUtils.assertBatchLen(batch, 4)
        outputs = []
        for input, do in zip(batch.highResRGBs(), mask):
            outputs.append(self.predictOneSide(input) if do else None)

        return outputs

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predictOneSide(self, imgL):
        with torch.no_grad():
            output = P.data_parallel(self.model, imgL * self.args.rgb_range)
            output = myUtils.quantize(output, self.args.rgb_range) / self.args.rgb_range
            return output

    def test(self, batch, type='l1', returnOutputs=False):
        myUtils.assertBatchLen(batch, 8)

        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        outputsIm = self.predict(batch.lastScaleBatch(), mask=[gt is not None for gt in batch.highResRGBs()])
        for gt, output, side in zip(batch.highResRGBs(), outputsIm, ('L', 'R')):
            scores[type + side] = evalFcn.getEvalFcn(type)(
                gt * self.args.rgb_range, output * self.args.rgb_range
            )if output is not None else None
            outputs['output' + side] = output

        return scores, outputs

    def load(self, checkpointDir):
        checkpointDir = super(SR, self).beforeLoad(checkpointDir)
        if checkpointDir is None:
            return

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
        super(SR, self).beforeSave(epoch, iteration)

        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
        }, self.checkpointDir)

class SRdisp(SR):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, cInput=6, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SRdisp, self).__init__(cInput, cuda, half, stage, dataset, saveFolderSuffix)

    def initModel(self):
        super(SRdisp, self).initModel()

    def warpAndCat(self, batch):
        inputL, inputR, dispL, dispR = batch
        with torch.no_grad():
            warpToL, warpToR, maskL, maskR = warp(*batch)
            warpTos = (warpToL, warpToR)
            cated = []
            for input in zip((inputL, inputR), (warpToL, warpToR), (maskL, maskR)):
                if self.args.n_inputs == 7:
                    cated.append(torch.cat(input, 1))
                elif self.args.n_inputs == 6:
                    cated.append(torch.cat(input[:2], 1))
                else:
                    raise Exception(
                        'Error: self.model.args.n_inputs = %d which is not supporty!' % self.model.args.n_inputs)
            return cated, warpTos

    def train(self, batch, returnOutputs=False):
        myUtils.assertBatchLen(batch, 8)
        cated, warpTos = self.warpAndCat(batch.lastScaleBatch())
        batch.lowResRGBs(cated)
        losses, outputs = super(SRdisp, self).train(batch, returnOutputs)
        if returnOutputs:
            for warpTo, side in zip(warpTos, ('L', 'R')):
                outputs['warpTo' + side] = warpTo
        return losses, outputs

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
        myUtils.assertBatchLen(batch, 4)
        cated, warpTos = self.warpAndCat(batch.firstScaleBatch())
        batch.highResRGBs(cated)
        return super(SRdisp, self).predict(batch, mask)

