import torch.utils.data as data
import random
from PIL import Image
import numpy as np
from utils import preprocess
from utils import python_pfm as pfm
import torch


def rgbLoader(path):
    return Image.open(path).convert('RGB')


def pfmLoader(path):
    return pfm.readPFM(path)[0]


def grayLoader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, inputLdirs, inputRdirs, gtLdirs=None, gtRdirs=None, training=False,
                 trainCrop=(512, 256), kitti=False):
        self.inputLdirs = inputLdirs
        self.inputRdirs = inputRdirs
        self.gtLdirs = gtLdirs
        self.gtRdirs = gtRdirs
        self.inputLoader = rgbLoader
        self.gtLoader = grayLoader if kitti else pfmLoader
        self.training = training
        self.trainCrop = trainCrop
        self.testCrop = (1232, 368) if kitti else None
        self.dispScale = 256 if kitti else 1

    def __getitem__(self, index):
        inputLdir = self.inputLdirs[index]
        inputRdir = self.inputRdirs[index]
        inputL = self.inputLoader(inputLdir)
        inputR = self.inputLoader(inputRdir)

        gtLdir = self.gtLdirs[index] if self.gtLdirs is not None else None
        gtRdir = self.gtRdirs[index] if self.gtRdirs is not None else None
        gtL = self.gtLoader(gtLdir) if gtLdir is not None else None
        gtR = self.gtLoader(gtRdir) if gtRdir is not None else None

        if self.training:
            w, h = inputL.size
            wCrop, hCrop = self.trainCrop

            x1 = random.randint(0, w - wCrop)
            y1 = random.randint(0, h - hCrop)

            inputL = inputL.crop((x1, y1, x1 + wCrop, y1 + hCrop))
            inputR = inputR.crop((x1, y1, x1 + wCrop, y1 + hCrop))

            try:
                gtL = gtL.crop((x1, y1, x1 + wCrop, y1 + hCrop)) if gtL is not None else None
                gtR = gtR.crop((x1, y1, x1 + wCrop, y1 + hCrop)) if gtR is not None else None
            except AttributeError:
                gtL = gtL[y1:y1 + hCrop, x1:x1 + wCrop] if gtL is not None else None
                gtR = gtR[y1:y1 + hCrop, x1:x1 + wCrop] if gtR is not None else None


        else:
            if self.testCrop is not None:
                w, h = inputL.size
                wCrop, hCrop = self.testCrop
                inputL = inputL.crop((w - wCrop, h - hCrop, w, h))
                inputR = inputR.crop((w - wCrop, h - hCrop, w, h))
                try:
                    gtL = gtL.crop((w - wCrop, h - hCrop, w, h)) if gtL is not None else None
                    gtR = gtR.crop((w - wCrop, h - hCrop, w, h)) if gtR is not None else None
                except AttributeError:
                    gtL = gtL[h:h + hCrop, w:w + wCrop] if gtL is not None else None
                    gtR = gtR[h:h + hCrop, w:w + wCrop] if gtR is not None else None

        processed = preprocess.get_transform(augment=False)
        inputL = processed(inputL)
        inputR = processed(inputR)

        gtL = np.ascontiguousarray(gtL, dtype=np.float32) / self.dispScale if gtL is not None else np.array([])
        gtR = np.ascontiguousarray(gtR, dtype=np.float32) / self.dispScale if gtR is not None else np.array([])

        return inputL, inputR, gtL, gtR

    def __len__(self):
        return len(self.inputLdirs)
