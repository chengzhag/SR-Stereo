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
                 trainCrop=(256, 512), kitti=False, loadScale=(1,), mode='training', mask=(1, 1, 1, 1)):
        self.mask = mask
        self.mode = mode
        self.dirs = (inputLdirs, inputRdirs, gtLdirs, gtRdirs)
        self.inputLoader = rgbLoader
        self.gtLoader = grayLoader if kitti else pfmLoader
        self.trainCrop = trainCrop
        self.testCrop = (round(1232 * loadScale[0]), round(368 * loadScale[0])) if kitti else None
        self.dispScale = 256 if kitti else 1
        self.loadScale = loadScale
        self.trainCrop = trainCrop


    def __getitem__(self, index):
        def scale(im, method, scaleRatios):
            w, h = im.size
            ims = []
            for r in scaleRatios:
                ims.append(im.resize((round(w * r), round(h * r)), method))
            return ims

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

        def loadIm(dirsIndex, loader, scaleRatios, isRGBorDepth):
            ims = []
            if not self.mask[dirsIndex] or self.dirs[dirsIndex] is None:
                return [np.array([]),] * len(self.loadScale)
            im0 = loader(self.dirs[dirsIndex][index])
            if type(im0) == np.ndarray:
                im0 = Image.fromarray(im0)

            # scale first to reduce time consumption
            scaleMethod = Image.ANTIALIAS if isRGBorDepth else Image.NEAREST
            im0 = scale(im0, scaleMethod, (scaleRatios[0],))[0]
            ims.append(im0)

            multiScales = []
            if len(scaleRatios) > 1:
                for i in range(1, len(scaleRatios)):
                    multiScales.append(scaleRatios[i] / scaleRatios[0])

            if self.mode == 'PIL':
                pass
            else:
                if self.mode == 'rawScaledTensor':
                    # scale to different sizes specified by scaleRatios
                    ims += scale(ims[0], scaleMethod, multiScales)
                    ims = [transforms.ToTensor()(im) for im in ims]
                elif self.mode in ('training', 'testing', 'submission'):
                    if self.mode == 'training':
                        # random crop
                        ims[0] = randomCrop(ims[0])
                    elif self.mode == 'testing':
                        if self.testCrop is not None:
                            # crop to the same size
                            ims[0] = testCrop(ims[0])
                    elif self.mode == 'submission':
                        # do no crop
                        pass
                    else:
                        raise Exception('No stats \'%s\'' % self.mode)
                    # scale to different sizes specified by scaleRatios
                    ims += scale(ims[0], scaleMethod, multiScales)
                    if isRGBorDepth:
                        processed = preprocess.get_transform(augment=False)
                        ims = [processed(im) for im in ims]
                else:
                    raise Exception('No mode %s!' % self.mode)

            if not isRGBorDepth:
                ims = [np.ascontiguousarray(im, dtype=np.float32) / self.dispScale * scaleRatio
                       for im, scaleRatio in zip(ims, scaleRatios)]
            return ims

        inputL = loadIm(0, self.inputLoader, self.loadScale, True)
        inputR = loadIm(1, self.inputLoader, self.loadScale, True)

        gtL = loadIm(2, self.gtLoader, self.loadScale, False)
        gtR = loadIm(3, self.gtLoader, self.loadScale, False)

        r = [im for scale in zip(inputL, inputR, gtL, gtR) for im in scale]

        return tuple(r)

    def __len__(self):
        for dirs in self.dirs:
            if dirs is not None:
                return len(dirs)
        raise Exception('Empty dataloader!')

    def name(self, index):
        for dirs in self.dirs:
            if dirs is not None:
                return dirs[index].split('/')[-1]
        raise Exception('Empty dataloader!')
