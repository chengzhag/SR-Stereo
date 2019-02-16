import torch
import os
import argparse

class NameValues:
    def __init__(self, prefix, suffixes, values):
        self._pairs = []
        for suffix, value in zip(suffixes, values):
            if value is not None:
                self._pairs.append((prefix + suffix, value))

    def str(self, unit=''):
        str = ''
        for name, value in self._pairs:
            str += '%s: %.2f%s, ' % (name, value, unit)
        return str

    def dic(self):
        dic={}
        for name, value in self._pairs:
            dic[name] = value
        return dic

    def pairs(self):
        return self._pairs

class AutoPad:
    def __init__(self, imgs, multiple):
        self.N, self.C, self.H, self.W = imgs.size()
        self.HPad = ((self.H - 1) // multiple + 1) * multiple
        self.WPad = ((self.W - 1) // multiple + 1) * multiple

    def pad(self, imgs, cuda):
        imgsPad = torch.zeros([self.N, self.C, self.HPad, self.WPad], dtype=imgs.dtype, device='cuda' if cuda else 'cpu')
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

# Log First n disparity maps into tensorboard
# Log All disparity maps if n == 0
def logFirstNdis(writer, name, disp, maxdisp, global_step=None, n=0):
    if disp is not None and n > 0:
        n = min(n, disp.size(0))
        disp = disp[:n, :, :]
        disp[disp > maxdisp] = maxdisp
        disp[disp < 0] = 0
        disp = disp / maxdisp
        disp = gray2rgb(disp)
        writer.add_images(name, disp, global_step=global_step)

def gray2rgb(im):
    return im.unsqueeze(1).repeat(1, 3, 1, 1)

def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def getBasicParser():
    parser = argparse.ArgumentParser(description='Stereo')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='PSMNet',
                        help='select model')
    parser.add_argument('--datapath', default='../datasets/sceneflow/',
                        help='datapath')
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval_fcn', type=str, default='l1',
                        help='evaluation function used in testing')
    parser.add_argument('--ndis_log', type=int, default=1,
                        help='number of disparity maps to log')
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        help='evaluation function used in testing')
    parser.add_argument('--load_scale', type=float, default=1,
                        help='scaling applied to data during loading')
    parser.add_argument('--crop_scale', type=float, default=None,
                        help='scaling applied to data during croping')
    parser.add_argument('--batchsize_train', type=int, default=6,
                        help='training batch size')
    parser.add_argument('--batchsize_test', type=int, default=6,
                        help='testing batch size')
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
