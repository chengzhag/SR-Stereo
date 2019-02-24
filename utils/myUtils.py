import torch
import os
import argparse
from tensorboardX import SummaryWriter


class NameValues:
    def __init__(self, names, values, prefix='', suffix=''):
        self._pairs = []
        self._names = []
        self._values = []
        for name, value in zip(names, values):
            if value is not None:
                self._pairs.append((prefix + name + suffix, value))
                self._names.append(name)
                self._values.append(value)

    def strPrint(self, unit=''):
        str = ''
        for name, value in self._pairs:
            str += '%s: ' % (name)
            if hasattr(value, '__iter__'):
                for v in value:
                    str += '%.2f%s, ' % (v, unit)
            else:
                str += '%.2f%s, ' % (value, unit)

        return str

    def strSuffix(self):
        str = ''
        for name, value in self._pairs:
            str += '_%s' % (name)
            if hasattr(value, '__iter__'):
                for v in value:
                    str += '_%.0f' % (v)
            else:
                str += '_%.0f' % (value)
        return str

    def dic(self):
        dic = {}
        for name, value in self._pairs:
            dic[name] = value
        return dic

    def pairs(self):
        return self._pairs

    def values(self):
        return self._values

    def names(self):
        return self._names


class AutoPad:
    def __init__(self, imgs, multiple):
        self.N, self.C, self.H, self.W = imgs.size()
        self.HPad = ((self.H - 1) // multiple + 1) * multiple
        self.WPad = ((self.W - 1) // multiple + 1) * multiple

    def pad(self, imgs, cuda):
        imgsPad = torch.zeros([self.N, self.C, self.HPad, self.WPad], dtype=imgs.dtype,
                              device='cuda' if cuda else 'cpu')
        imgsPad[:, :, (self.HPad - self.H):, (self.WPad - self.W):] = imgs
        return imgsPad

    def unpad(self, imgs):
        imgs = imgs[:, (self.HPad - self.H):, (self.WPad - self.W):]
        return imgs


# Flip among W dimension. For NCHW data type.
def flipLR(im):
    return im.flip(-1)


def assertDisp(dispL=None, dispR=None):
    if (dispL is None or dispL.numel() == 0) and (dispR is None or dispR.numel() == 0):
        raise Exception('No disp input!')


# Log First n ims into tensorboard
# Log All ims if n == 0
def logFirstNIms(writer, name, im, range, global_step=None, n=0):
    if im is not None:
        n = min(n, im.size(0))
        if n > 0:
            im = im[:n]
        if im.dim() == 3 or (im.dim() == 4 and im.size(1) == 1):
            im[im > range] = range
            im[im < 0] = 0
            im = im / range
            im = gray2rgb(im)
        writer.add_images(name, im, global_step=global_step)


def gray2rgb(im):
    if im.dim() == 3:
        im = im.unsqueeze(1)
    return im.repeat(1, 3, 1, 1)


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getBasicParser(includeKeys=['all'], description='Stereo'):
    parser = argparse.ArgumentParser(description=description)

    addParams = {'maxdisp': lambda: parser.add_argument('--maxdisp', type=int, default=192,
                                                        help='maxium disparity of dataset (before scaling)'),
                 'dispscale': lambda: parser.add_argument('--dispscale', type=float, default=1,
                                                          help='scale disparity when training and predicting (real disparity range of stereo net will be set as maxdisp/dispscale)'),
                 'model': lambda: parser.add_argument('--model', default='PSMNet',
                                                      help='select model'),
                 'datapath': lambda: parser.add_argument('--datapath', default='../datasets/sceneflow/',
                                                         help='datapath'),
                 'loadmodel': lambda: parser.add_argument('--loadmodel', default=None,
                                                          help='load model'),
                 'no_cuda': lambda: parser.add_argument('--no_cuda', action='store_true', default=False,
                                                        help='enables CUDA training'),
                 'seed': lambda: parser.add_argument('--seed', type=int, default=1, metavar='S',
                                                     help='random seed (default: 1)'),
                 'eval_fcn': lambda: parser.add_argument('--eval_fcn', type=str, default='outlier',
                                                         help='evaluation function used in testing'),
                 'ndis_log': lambda: parser.add_argument('--ndis_log', type=int, default=1,
                                                         help='number of disparity maps to log'),
                 'dataset': lambda: parser.add_argument('--dataset', type=str, default='sceneflow',
                                                        help='(sceneflow/kitti2012/kitti2015/carla_kitti)'),
                 'load_scale': lambda: parser.add_argument('--load_scale', type=float, default=1,
                                                           help='scaling applied to data during loading'),
                 'trainCrop': lambda: parser.add_argument('--trainCrop', type=float, default=(256, 512), nargs=2,
                                                          help='size of random crop (H x W) applied to data during training'),
                 'batchsize_test': lambda: parser.add_argument('--batchsize_test', type=int, default=3,
                                                               help='testing batch size'),
                 # training
                 'batchsize_train': lambda: parser.add_argument('--batchsize_train', type=int, default=3,
                                                                help='training batch size'),
                 'log_every': lambda: parser.add_argument('--log_every', type=int, default=10,
                                                          help='log every log_every iterations. set to 0 to stop logging'),
                 'test_every': lambda: parser.add_argument('--test_every', type=int, default=1,
                                                           help='test every test_every epochs. set to 0 to stop testing'),
                 'epochs': lambda: parser.add_argument('--epochs', type=int, default=10,
                                                       help='number of epochs to train'),
                 'lr': lambda: parser.add_argument('--lr', type=float, default=[0.001], help='', nargs='+'),
                 # submission
                 'subtype': lambda: parser.add_argument('--subtype', type=str, default='eval',
                                                        help='dataset type used for submission (eval/test)'),
                 # module test
                 'nsample_save': lambda: parser.add_argument('--nsample_save', type=int, default=5,
                                                             help='save n samples in module testing'),
                 # half precision
                 'half': lambda: parser.add_argument('--half', action='store_true', default=False,
                                                     help='enables half precision'),
                 # SRdisp specified param
                 'withMask': lambda:  parser.add_argument('--withMask', action='store_true', default=False,
                                                     help='input 7 channels with mask to SRdisp instead of 6'),
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


def assertMode(kitti, mode):
    if kitti:
        print(
            'Using dataset KITTI. Evaluation will exclude zero disparity pixels. And only left disparity map will be considered.')
        return 'left'
    else:
        if mode not in ('left', 'right', 'both'):
            raise Exception('No mode \'%s!\'' % mode)
        return mode


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

class TensorboardLogger:
    def __init__(self):
        self.writer = None

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def init(self, folder):
        if self.writer is None:
            self.writer = SummaryWriter(folder)
        
    def logFirstNIms(self, name, im, range, global_step=None, n=0):
        if self.writer is None:
            raise Exception('Error: SummaryWriter is not initialized!')
        logFirstNIms(self.writer, name, im, range, global_step, n)


