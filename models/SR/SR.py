import torch.optim as optim
import torch
import torch.nn.functional as F
from evaluation import evalFcn
from utils import myUtils
from .RawEDSR import edsr
from ..Model import Model
import torch.nn.parallel as P
import collections

class RawEDSR(edsr.EDSR):
    def __init__(self, cInput):
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
        super(RawEDSR, self).__init__(self.args)

    # input: RGB value range 0~1
    # output: RGB value range 0~1 without quantize
    def forward(self, imgL):
        output = P.data_parallel(super(RawEDSR, self).forward, imgL * self.args.rgb_range) / self.args.rgb_range
        return output

class SR(Model):
    # dataset: only used for suffix of saveFolderName
    def __init__(self, cInput=3, cuda=True, half=False, stage='unnamed', dataset=None, saveFolderSuffix=''):
        super(SR, self).__init__(cuda, half, stage, dataset, saveFolderSuffix)
        self.cInput = cInput
        self.getModel = RawEDSR

    def initModel(self):
        self.model = RawEDSR(self.cInput)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model.cuda()

    # outputs, gts: RGB value range 0~1
    def loss(self, outputs, gts):
        # To get same loss with orignal EDSR, input range should scale to 0~self.args.rgb_range
        loss = F.smooth_l1_loss(outputs * self.model.args.rgb_range, gts * self.model.args.rgb_range, reduction='mean')
        return loss

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
        output = self.model.forward(imgL)
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
            output = self.model.forward(imgL)
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
                gt * self.model.args.rgb_range, output * self.model.args.rgb_range
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

    def save(self, epoch, iteration, trainLoss):
        self.savePrepare(epoch, iteration)
        saveDict = {
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'train_loss': trainLoss,
        }
        if self.optimizer is not None:
            saveDict['optimizer'] = self.optimizer.state_dict()
        torch.save(saveDict, self.checkpointDir)

