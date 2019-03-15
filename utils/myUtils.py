import torch
import os
import argparse
from tensorboardX import SummaryWriter
import collections
import cv2
import numpy as np
import torchvision.transforms as transforms
import random


class NameValues(collections.OrderedDict):
    def __init__(self, names=(), values=(), prefix='', suffix=''):
        super(NameValues, self).__init__()
        for name, value in zip(names, values):
            if value is not None:
                super(NameValues, self).__setitem__(prefix + name + suffix, value)

    def strPrint(self, prefix='', suffix=''):
        strReturn = ''
        for name, value in super(NameValues, self).items():
            if name.find('outlier') != -1:
                unit = '%'
            else:
                unit = ''
            strReturn += '%s: ' % (prefix + name + suffix)
            def addValue(value):
                s = ''
                if type(value) in (list, tuple):
                    for v in value:
                        s += addValue(v)
                else:
                    s += '%.3f%s, ' % (value, unit)
                return s

            strReturn += addValue(value)

        return strReturn

    def strSuffix(self, prefix='', suffix=''):
        sSuffix = ''
        for name, value in super(NameValues, self).items():
            sSuffix += '_%s' % (prefix + name + suffix)

            def addValue(sAppend, values):
                if type(values) == int:
                    return sAppend + '_' + str(values)
                elif type(values) == float:
                    return sAppend + '_%.1f' % values
                elif type(values) in (list, tuple):
                    for v in values:
                        sAppend = addValue(sAppend, v)
                    return sAppend
                else:
                    raise Exception('Error: Type of values should be in int, float, list, tuple!')

            sSuffix = addValue(sSuffix, value)

        return sSuffix



class AutoPad:
    def __init__(self, imgs, multiple):
        self.N, self.C, self.H, self.W = imgs.size()
        self.HPad = ((self.H - 1) // multiple + 1) * multiple
        self.WPad = ((self.W - 1) // multiple + 1) * multiple

    def pad(self, imgs):
        def _pad(img):
            imgPad = torch.zeros([self.N, self.C, self.HPad, self.WPad], dtype=img.dtype,
                                  device=img.device.type)
            imgPad[:, :, (self.HPad - self.H):, (self.WPad - self.W):] = img
            return imgPad
        return forNestingList(imgs, _pad)

    def unpad(self, imgs):
        return forNestingList(imgs, lambda img: img[:, (self.HPad - self.H):, (self.WPad - self.W):])


# Flip among W dimension. For NCHW data type.
def flipLR(ims):
    return forNestingList(ims, lambda im: im.flip(-1) if im is not None else None)

def assertDisp(dispL=None, dispR=None):
    if (dispL is None or dispL.numel() == 0) and (dispR is None or dispR.numel() == 0):
        raise Exception('No disp input!')


# Log First n ims into tensorboard
# Log All ims if n == 0
def logFirstNIms(writer, name, im, range, global_step=None, n=0):
    if im is not None:
        n = min(n, im.size(0))
        if n > 0 and im.dim() > 2:
            im = im[:n]
        if im.dim() == 3 or im.dim() == 2 or (im.dim() == 4 and im.size(1) == 1):
            im[im > range] = range
            im[im < 0] = 0
            im = im / range
            im = gray2rgb(im.cpu())
        writer.add_images(name, im, global_step=global_step)


def gray2rgb(im):
    if im.dim() == 2:
        im = im.unsqueeze(0)
    if im.dim() == 3:
        im = im.unsqueeze(1)
    return im.repeat(1, 3, 1, 1)

def gray2color(im):
    if im.dim() == 4 and im.size(1) == 1:
        im = im.squeeze(1)
    if im.dim() == 3 and im.size(0) >= 1:
        imReturn = torch.zeros([im.size(0), 3, im.size(1), im.size(2)], dtype=torch.uint8)
        for i in range(im.size(0)):
            imReturn[i, :, :, :] = gray2color(im[i, :, :])
        return imReturn
    elif im.dim() == 2:
        im = (im.numpy() * 255).astype(np.uint8)
        im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        im = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
        return im
    else:
        raise Exception('Error: Input of gray2color must have one channel!')



def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getBasicParser(includeKeys=['all'], description='Stereo'):
    parser = argparse.ArgumentParser(description=description)

    addParams = {'seed': lambda: parser.add_argument('--seed', type=int, default=1, metavar='S',
                                                     help='random seed (default: 1)'),
                 # model
                 'outputFolder': lambda: parser.add_argument('--outputFolder', type=str, default=None,
                                                             help='output checkpoints and logs to foleder logs/outputFolder'),
                 'maxdisp': lambda: parser.add_argument('--maxdisp', type=int, default=192,
                                                        help='maximum disparity of unscaled model (or dataset in some module test)'),
                 'dispscale': lambda: parser.add_argument('--dispscale', type=float, default=1,
                                                          help='scale disparity when training (gtDisp/dispscale) and predicting (outputDisp*dispscale'),
                 'model': lambda: parser.add_argument('--model', default='PSMNet',
                                                      help='select model'),
                 'loadmodel': lambda: parser.add_argument('--loadmodel',  type=str, default=None, nargs='+',
                                                          help='checkpoint(s) of model(s) to load'),
                 'no_cuda': lambda: parser.add_argument('--no_cuda', action='store_true', default=False,
                                                        help='enables CUDA training'),
                 # logging
                 'ndis_log': lambda: parser.add_argument('--ndis_log', type=int, default=1,
                                                         help='number of disparity maps to log'),
                 # datasets
                 'dataset': lambda: parser.add_argument('--dataset', type=str, default='sceneflow',
                                                        help='(sceneflow/kitti2012/kitti2015/carla_kitti)'),
                 'datapath': lambda: parser.add_argument('--datapath', default='../datasets/sceneflow/',
                                                         help='datapath'),
                 'load_scale': lambda: parser.add_argument('--load_scale', type=float, default=[1], nargs='+',
                                                           help='scaling applied to data during loading'),
                 'randomLR': lambda: parser.add_argument('--randomLR', type=str, default=None,
                                                        help='enables randomly loading left or right images (disp/rgb)'),
                 # training
                 'batchsize_train': lambda: parser.add_argument('--batchsize_train', type=int, default=4,
                                                                help='training batch size'),
                 'trainCrop': lambda: parser.add_argument('--trainCrop', type=int, default=(256, 512), nargs=2,
                                                          help='size of random crop (H x W) applied to data during training'),
                 'log_every': lambda: parser.add_argument('--log_every', type=int, default=10,
                                                          help='log every log_every iterations. set to 0 to stop logging'),
                 'save_every': lambda: parser.add_argument('--save_every', type=int, default=1,
                                                          help='save every save_every epochs; '
                                                               'set to -1 to train without saving; '
                                                               'set to 0 to save after the last epoch.'),
                 'test_every': lambda: parser.add_argument('--test_every', type=int, default=1,
                                                           help='test every test_every epochs. '
                                                                '> 0 will not test before training. '
                                                                '= 0 will test before training and after final epoch. '
                                                                '< 0 will test before training'),
                 'epochs': lambda: parser.add_argument('--epochs', type=int, default=10,
                                                       help='number of epochs to train'),
                 'lr': lambda: parser.add_argument('--lr', type=float, default=[0.001], help='', nargs='+'),
                 'lossWeights': lambda: parser.add_argument('--lossWeights', type=float, default=[1], nargs='+',
                                                           help='weights of losses if model have multiple losses'),
                 'resume': lambda: parser.add_argument('--resume', action='store_true', default=False,
                                                           help='resume specified training '
                                                                '(or save evaluation results to old folder)'
                                                                ' else save/log into a new folders'),
                 # evaluation
                 'eval_fcn': lambda: parser.add_argument('--eval_fcn', type=str, default='outlier',
                                                         help='evaluation function used in testing'),
                 'batchsize_test': lambda: parser.add_argument('--batchsize_test', type=int, default=4,
                                                               help='testing batch size'),
                 'subValidSet': lambda: parser.add_argument('--subValidSet', type=float, default=1,
                                                               help='test with part of valid set'),
                 # submission
                 'subtype': lambda: parser.add_argument('--subtype', type=str, default='subEval',
                                                        help='dataset type used for submission (eval/test)'),
                 # module test
                 'nsample_save': lambda: parser.add_argument('--nsample_save', type=int, default=1,
                                                             help='save n samples in module testing'),
                 # half precision
                 'half': lambda: parser.add_argument('--half', action='store_true', default=False,
                                                     help='enables half precision'),
                 # SRdisp specified param
                 'withMask': lambda: parser.add_argument('--withMask', action='store_true', default=False,
                                                         help='input 7 channels with mask to SRdisp instead of 6'),
                 # SRdispStereoRefine specified param
                 'itRefine': lambda: parser.add_argument('--itRefine', type=int, default=1,
                                                             help='iterations of refining process'),
                 }

    if len(includeKeys):
        if includeKeys[0] == 'all':
            for addParam in addParams.values():
                addParam()
        else:
            for key in includeKeys:
                addParams[key]()

    return parser


def adjustLearningRate(optimizer, epoch, lr):
    if len(lr) % 2 == 0:
        raise Exception('lr setting should be like \'0.001 300 0.0001 \'')
    nThres = len(lr) // 2 + 1
    for iThres in range(nThres):
        lrThres = lr[2 * iThres]
        if iThres < nThres - 1:
            epochThres = lr[2 * iThres + 1]
            if epoch <= epochThres:
                lr = lrThres
                break
        else:
            lr = lrThres
    print('lr = %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def assertBatchLen(batch, length):
    if type(batch) is not Batch:
        raise Exception('Error: batch must be class Batch!')
    if type(length) in (list, tuple):
        if len(batch) not in length:
            raise Exception(f'Error: input batch with length {len(batch)} doesnot match required {length}!')
    elif len(batch) != length:
        raise Exception(f'Error: input batch with length {len(batch)} doesnot match required {length}!')


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


class TensorboardLogger:
    def __init__(self):
        self.writer = None
        self._folder = None

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def set(self, folder):
        if self.writer is None:
            self.writer = SummaryWriter(folder)
        else:
            if folder != self._folder:
                self.writer.close()
                self.writer = SummaryWriter(folder)
        self._folder = folder

    def logFirstNIms(self, name, im, range, global_step=None, n=0):
        if self.writer is None:
            raise Exception('Error: SummaryWriter is not initialized!')
        logFirstNIms(self.writer, name, im, range, global_step, n)


class Batch:
    def __init__(self, batch, cuda=False, half=False):
        if type(batch) in (list, tuple):
            self._assertLen(len(batch))
            self.batch = batch[:]  # deattach with initial list
        elif type(batch) is Batch:
            self._assertLen(len(batch))
            self.batch = batch.batch[:]
        elif type(batch) is int:
            self._assertLen(batch)
            if batch % 4 != 0:
                raise Exception(f'Error: input batch with length {len(batch)} doesnot match required 4n!')
            self.batch = [None] * batch
        else:
            raise Exception('Error: batch must be class list, tuple or Batch!')

        self.half = half
        self.cuda = cuda
        self.batch = [(im if im.numel() else None) if im is not None else None for im in self.batch]
        if half:
            self.batch = [(im.half() if half else im) if im is not None else None for im in self.batch]
        if cuda:
            self.batch = [(im.cuda() if cuda else im) if im is not None else None for im in self.batch]

        def assertData(t):
            if t is not None and torch.isnan(t).any():
                raise Exception('Error: Data has nan in it')

        forNestingList(self.batch, assertData)

    def _assertLen(self, len):
        if len % 4 != 0:
            raise Exception(f'Error: input batch with length {len} doesnot match required 4n!')

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, item):
        return self.batch[item]

    def __setitem__(self, key, value):
        self.batch[key] = value

    def detach(self):
        return Batch(self, cuda=self.cuda, half=self.half)

    def lastScaleBatch(self):
        return Batch(self.batch[-4:], cuda=self.cuda, half=self.half)

    def firstScaleBatch(self):
        return Batch(self.batch[:4], cuda=self.cuda, half=self.half)

    def highResRGBs(self, set=None):
        if set is not None:
            self.batch[0:2] = set
        return self.batch[0:2]

    def highResDisps(self, set=None):
        if set is not None:
            self.batch[2:4] = set
        return self.batch[2:4]

    def lowResRGBs(self, set=None):
        if set is not None:
            self.batch[4:6] = set
        return self.batch[4:6]

    def lowResDisps(self, set=None):
        if set is not None:
            self.batch[6:8] = set
        return self.batch[6:8]

    def lowestResRGBs(self, set=None):
        if set is not None:
            self.batch[-4:-2] = set
        return self.batch[-4:-2]

    def lowestResDisps(self, set=None):
        if set is not None:
            self.batch[-2:] = set
        return self.batch[-2:]

    def allRGBs(self, set=None):
        if set is not None:
            self.batch[0::4] = set[:len(set) // 2]
            self.batch[1::4] = set[len(set) // 2:]
        return self.batch[0::4] + self.batch[1::4]

    def allDisps(self, set=None):
        if set is not None:
            self.batch[2::4] = set[:len(set) // 2]
            self.batch[3::4] = set[len(set) // 2:]
        return self.batch[2::4] + self.batch[3::4]

def forNestingList(l, fcn):
    if type(l) in (list, tuple):
        l = [forNestingList(e, fcn) for e in l]
        return l
    else:
        return fcn(l)

def getLastNotList(l):
    if type(l) in (list, tuple):
        return getLastNotList(l[-1])
    else:
        return l
def scanCheckpoint(checkpointDirs):
    if type(checkpointDirs) in (list, tuple):
        checkpointDirs = [scanCheckpoint(dir) for dir in checkpointDirs]
    else:
        # if checkpoint is folder
        if os.path.isdir(checkpointDirs):
            filenames = [d for d in os.listdir(checkpointDirs) if os.path.isfile(os.path.join(checkpointDirs, d))]
            filenames.sort()
            latestCheckpointName = None
            latestEpoch = None

            def _getEpoch(name):
                try:
                    keywords = name.split('_')
                    epoch = keywords[keywords.index('epoch') + 1]
                    return int(epoch)
                except ValueError:
                    return None

            for filename in filenames:
                if any(filename.endswith(extension) for extension in ('.tar', '.pt')):
                    if latestCheckpointName is None:
                        latestCheckpointName = filename
                        latestEpoch = _getEpoch(filename)
                    else:
                        epoch = _getEpoch(filename)
                        if epoch > latestEpoch or epoch is None:
                            latestCheckpointName = filename
                            latestEpoch = epoch
            checkpointDirs = os.path.join(checkpointDirs, latestCheckpointName)

    return checkpointDirs

def getSuffix(checkpointDirOrFolder):
    # saveFolderSuffix = myUtils.NameValues(('loadScale', 'trainCrop', 'batchSize', 'lossWeights'),
    #                                       (trainImgLoader.loadScale,
    #                                        trainImgLoader.trainCrop,
    #                                        args.batchsize_train,
    #                                        args.lossWeights))
    if type(checkpointDirOrFolder) is str or \
            (type(checkpointDirOrFolder) in (list, tuple) and len(checkpointDirOrFolder) == 1):
        checkpointDir = scanCheckpoint(checkpointDirOrFolder[0])
        checkpointFolder, _ = os.path.split(checkpointDir)
        checkpointFolder = checkpointFolder.split('/')[-1]
        saveFolderSuffix = checkpointFolder.split('_')[2:]
        saveFolderSuffix = ['_' + suffix for suffix in saveFolderSuffix]
        saveFolderSuffix = ''.join(saveFolderSuffix)
    else:
        saveFolderSuffix = ''
    return saveFolderSuffix

def depth(l):
    if type(l) in (tuple, list):
        return 1 + max(depth(item) for item in l)
    else:
        return 0

class Filter:
    def __init__(self, weight=0.1):
        self.weight = weight
        self.old = None
    def __call__(self, x):
        self.old = x if self.old is None else self.old * (1 - self.weight) + x * self.weight
        return self.old

def savePreprocessRGB(im):
    output = im.squeeze()
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    output = (output * 255).astype('uint8')
    return output

def savePreprocessDisp(disp):
    dispOut = disp.squeeze()
    dispOut = dispOut.data.cpu().numpy()
    dispOut = (dispOut * 256).astype('uint16')
    return dispOut

def shuffleLists(lists):
    c = list(zip(*lists))
    random.shuffle(c)
    lists = list(zip(*c))
    return lists

