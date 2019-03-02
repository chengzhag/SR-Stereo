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

    # outputs, gts: RGB value range 0~1
    def loss(self, outputs, gts):
        # To get same loss with orignal EDSR, input range should scale to 0~self.args.rgb_range
        loss = F.smooth_l1_loss(outputs * self.args.rgb_range, gts * self.args.rgb_range, reduction='mean')
        return loss

    # input: RGB value range 0~1
    # output: RGB value range 0~1 without quantize
    def forward(self, imgL):
        output = P.data_parallel(self.model, imgL * self.args.rgb_range) / self.args.rgb_range
        return output

    # imgL: RGB value range 0~1
    # imgH: RGB value range 0~1
    def train(self, batch, returnOutputs=False):
        myUtils.assertBatchLen(batch, 8)
        self.trainPrepare()

        losses = myUtils.NameValues()
        outputs = collections.OrderedDict()
        for input, gt, side in zip(batch.lowResRGBs(), batch.highResRGBs(), ('L', 'R')):
            if gt is not None:
                loss, predict = self.trainOneSide(input, gt, returnOutputs)
                losses['loss' + side] = loss
                if returnOutputs:
                    outputs['output' + side] = predict

        return losses, outputs

    def trainOneSide(self, imgL, imgH, returnOutputs=False):
        self.optimizer.zero_grad()
        output = self.forward(imgL)
        loss = self.loss(imgH, output)
        with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
        output = myUtils.quantize(output.detach(), 1) if returnOutputs else None
        return loss.data.item(), output

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predict(self, batch, mask=(1,1)):
        myUtils.assertBatchLen(batch, 4)
        self.predictPrepare()

        outputs = []
        for input, do in zip(batch.lowestResRGBs(), mask):
            outputs.append(self.predictOneSide(input) if do else None)

        return outputs

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    def predictOneSide(self, imgL):
        with torch.no_grad():
            output = self.forward(imgL)
            output = myUtils.quantize(output, 1)
            return output

    def test(self, batch, type='l1', returnOutputs=False):
        myUtils.assertBatchLen(batch, 8)

        scores = myUtils.NameValues()
        outputs = collections.OrderedDict()
        mask = [gt is not None for gt in batch.highResRGBs()]
        outputsIm = self.predict(batch.lastScaleBatch(), mask=mask)
        for gt, output, side in zip(batch.highResRGBs(), outputsIm, ('L', 'R')):
            scores[type + side] = evalFcn.getEvalFcn(type)(
                gt * self.args.rgb_range, output * self.args.rgb_range
            )if output is not None else None
            if returnOutputs:
                outputs['output' + side] = output

        return scores, outputs

    def load(self, checkpointDir):
        checkpointDir = self.loadPrepare(checkpointDir)
        if checkpointDir is None:
            return None, None

        loadStateDict = torch.load(checkpointDir)
        if 'optimizer' in loadStateDict.keys():
            self.optimizer.load_state_dict(loadStateDict['optimizer'])
        loadModelDict = loadStateDict.get('state_dict', loadStateDict)

        newModelDict = self.model.state_dict()
        selectedModelDict = {}
        for loadName, loadValue in loadModelDict.items():
            if loadName in newModelDict and newModelDict[loadName].size() == loadValue.size():
                selectedModelDict[loadName] = loadValue
            else:
                # try to initialize input layer from weights with different channels
                # if loadName == 'head.0.weight':
                #     selectedModelDict[loadName] = newModelDict[loadName]
                #     selectedModelDict[loadName][:,:3,:,:] = loadValue
                #     selectedModelDict[loadName][:, 3:6, :, :] = loadValue
                print('Warning! While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(loadName, newModelDict[loadName].size(), loadValue.size()))
        newModelDict.update(selectedModelDict)
        self.model.load_state_dict(newModelDict, strict=False)
        print('Loading complete! Number of model parameters: %d' % self.nParams())

        epoch = loadStateDict.get('epoch')
        iteration = loadStateDict.get('iteration')
        print(f'Checkpoint epoch {epoch}, iteration {iteration}')
        return epoch, iteration

    def save(self, epoch, iteration, trainLoss, toOld=False):
        super(SR, self).savePrepare(epoch, iteration, toOld)
        saveDict = {
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
        }
        if self.optimizer is not None:
            saveDict['optimizer'] = self.optimizer.state_dict()
        torch.save(saveDict, self.checkpointDir)

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
        self.trainPrepare()

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
        self.predictPrepare()

        cated, warpTos = self.warpAndCat(batch.firstScaleBatch())
        batch.highResRGBs(cated)
        outputs = super(SRdisp, self).predict(batch, mask)
        return outputs

