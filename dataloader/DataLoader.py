import torch.utils.data as data
import random
from PIL import Image
import numpy as np
from utils import preprocess
from utils import python_pfm as pfm
import torchvision.transforms as transforms


def rgbLoader(path):
    return Image.open(path).convert('RGB')


def pfmLoader(path):
    return pfm.readPFM(path)[0]


def grayLoader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    # trainCrop = (W, H)
    def __init__(self, inputLdirs=None, inputRdirs=None, gtLdirs=None, gtRdirs=None,
                 trainCrop=(256, 512), kitti=False, loadScale=1, mode='training'):
        self.mode = mode
        self.inputLdirs = inputLdirs
        self.inputRdirs = inputRdirs
        # in submission, only input images are needed
        self.gtLdirs = gtLdirs
        self.gtRdirs = gtRdirs
        self.inputLoader = rgbLoader
        self.gtLoader = grayLoader if kitti else pfmLoader
        self.trainCrop = trainCrop
        self.testCrop = (round(1232 * loadScale), round(368 * loadScale)) if kitti else None
        self.dispScale = 256 if kitti else 1
        self.loadScale = loadScale
        self.trainCrop = trainCrop


    def __getitem__(self, index):
        def scale(im, method, scaleRatio):
            w, h = im.size
            return im.resize((round(w * scaleRatio), round(h * scaleRatio)), method)

        def testCrop(im):
            w, h = im.size
            wCrop, hCrop = self.testCrop
            return im.crop((w - wCrop, h - hCrop, w, h))

        class RandomCrop:
            def __init__(self, trainCrop):
                self.hCrop, self.wCrop = trainCrop
                self.x1 = None
                self.y1 = None

            def __call__(self, input):
                w, h = input.size
                if self.x1 is None: self.x1 = random.randint(0, w - self.wCrop)
                if self.y1 is None: self.y1 = random.randint(0, h - self.hCrop)
                return input.crop((self.x1, self.y1, self.x1 + self.wCrop, self.y1 + self.hCrop))

        randomCrop = RandomCrop(trainCrop=self.trainCrop)

        def loadIm(dirs, loader, scaleRatio, isRGBorDepth):
            if dirs is None: return np.array([])
            im = loader(dirs[index])
            if type(im) == np.ndarray:
                im = Image.fromarray(im)

            if self.mode == 'rawUnscaledTensor':
                im = transforms.ToTensor()(im) if im is not None else None
                return im

            im = scale(im, Image.ANTIALIAS, scaleRatio)

            if self.mode == 'PIL':
                pass
            elif self.mode == 'rawScaledTensor':
                im = transforms.ToTensor()(im) if im is not None else None
            elif self.mode in ('training', 'testing', 'submission'):
                if self.mode == 'training':
                    # random crop
                    im = randomCrop(im)
                elif self.mode == 'testing':
                    if self.testCrop is not None:
                        # crop to the same size
                        im = testCrop(im)
                elif self.mode == 'submission':
                    # do no crop
                    pass
                else:
                    raise Exception('No stats \'%s\'' % self.mode)

                if isRGBorDepth:
                    processed = preprocess.get_transform(augment=False)
                    im = processed(im)
            else:
                raise Exception('No mode %s!' % self.mode)

            if not isRGBorDepth:
                im = np.ascontiguousarray(im, dtype=np.float32) / self.dispScale * scaleRatio
            return im

        inputL = loadIm(self.inputLdirs, self.inputLoader, self.loadScale, True)
        inputR = loadIm(self.inputRdirs, self.inputLoader, self.loadScale, True)

        gtL = loadIm(self.gtLdirs, self.gtLoader, self.loadScale, False)
        gtR = loadIm(self.gtRdirs, self.gtLoader, self.loadScale, False)

        return inputL, inputR, gtL, gtR

    def __len__(self):
        return len(self.inputLdirs)
