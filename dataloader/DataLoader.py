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
    def __init__(self, inputLdirs, inputRdirs, gtLdirs=None, gtRdirs=None, training=False,
                 trainCrop=(512, 256), kitti=False, loadScale=1, cropScale=1, mode='normal'):
        self.inputLdirs = inputLdirs
        self.inputRdirs = inputRdirs
        self.gtLdirs = gtLdirs
        self.gtRdirs = gtRdirs
        self.inputLoader = rgbLoader
        self.gtLoader = grayLoader if kitti else pfmLoader
        self.training = training
        self.trainCrop = trainCrop
        self.testCrop = (round(1232 * loadScale), round(368 * loadScale)) if kitti else None
        self.dispScale = 256 if kitti else 1
        self.loadScale = loadScale
        self.cropScale = cropScale
        self.trainCrop = (round(trainCrop[0] * self.cropScale), round(trainCrop[1] * self.cropScale))
        self.mode = mode

    def __getitem__(self, index):
        def scale(im, method):
            w, h = im.size
            return im.resize((round(w * self.loadScale), round(h * self.loadScale)), method)

        inputLdir = self.inputLdirs[index]
        inputRdir = self.inputRdirs[index]
        inputL = self.inputLoader(inputLdir)
        inputR = self.inputLoader(inputRdir)

        gtLdir = self.gtLdirs[index] if self.gtLdirs is not None else None
        gtRdir = self.gtRdirs[index] if self.gtRdirs is not None else None
        gtL = self.gtLoader(gtLdir) if gtLdir is not None else None
        gtR = self.gtLoader(gtRdir) if gtRdir is not None else None
        if type(gtL) == np.ndarray or type(gtR) == np.ndarray:
            gtL = Image.fromarray(gtL) if gtLdir is not None else None
            gtR = Image.fromarray(gtR) if gtRdir is not None else None

        if self.mode == 'raw':
            inputL, inputR, gtL, gtR = [transforms.ToTensor()(im) if im is not None else None
                                        for im in (inputL, inputR, gtL, gtR)]
            gtL, gtR = gtL.squeeze(), gtR.squeeze()
            return inputL, inputR, gtL, gtR

        inputL = scale(inputL, Image.ANTIALIAS)
        inputR = scale(inputR, Image.ANTIALIAS)
        gtL = scale(gtL, Image.NEAREST) if gtLdir is not None else None
        gtR = scale(gtR, Image.NEAREST) if gtRdir is not None else None

        if self.mode == 'PIL':
            pass

        elif self.mode == 'scaled':
            inputL, inputR = [transforms.ToTensor()(im) if im is not None else None
                              for im in (inputL, inputR)]
        elif self.mode == 'normal':
            if self.training:
                w, h = inputL.size
                wCrop, hCrop = self.trainCrop

                x1 = random.randint(0, w - wCrop)
                y1 = random.randint(0, h - hCrop)

                inputL = inputL.crop((x1, y1, x1 + wCrop, y1 + hCrop))
                inputR = inputR.crop((x1, y1, x1 + wCrop, y1 + hCrop))

                gtL = gtL.crop((x1, y1, x1 + wCrop, y1 + hCrop)) if gtL is not None else None
                gtR = gtR.crop((x1, y1, x1 + wCrop, y1 + hCrop)) if gtR is not None else None

            else:
                if self.testCrop is not None:
                    w, h = inputL.size
                    wCrop, hCrop = self.testCrop
                    inputL = inputL.crop((w - wCrop, h - hCrop, w, h))
                    inputR = inputR.crop((w - wCrop, h - hCrop, w, h))

                    gtL = gtL.crop((w - wCrop, h - hCrop, w, h)) if gtL is not None else None
                    gtR = gtR.crop((w - wCrop, h - hCrop, w, h)) if gtR is not None else None

            processed = preprocess.get_transform(augment=False)
            inputL = processed(inputL)
            inputR = processed(inputR)

        else:
            raise Exception('No mode %s!' % self.mode)

        gtL = np.ascontiguousarray(gtL, dtype=np.float32) / self.dispScale * self.loadScale \
            if gtL is not None else np.array([])
        gtR = np.ascontiguousarray(gtR, dtype=np.float32) / self.dispScale * self.loadScale \
            if gtR is not None else np.array([])

        return inputL, inputR, gtL, gtR

    def __len__(self):
        return len(self.inputLdirs)
