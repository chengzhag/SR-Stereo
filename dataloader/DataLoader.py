import torch.utils.data as data
import random
from PIL import Image
import numpy as np
from utils import python_pfm as pfm
import torchvision.transforms as transforms
import operator


def rgbLoader(path):
    return Image.open(path).convert('RGB')


def pfmLoader(path):
    return pfm.readPFM(path)[0]


def grayLoader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    # trainCrop = (W, H)
    def __init__(self, inputLdirs=None, inputRdirs=None, gtLdirs=None, gtRdirs=None,
                 trainCrop=(256, 512), kitti=False, loadScale=(1,), mode='training',
                 mask=(1, 1, 1, 1), randomLR=None):
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
        self.randomLR = randomLR
        self.argument = kitti and mode == 'training' and operator.eq(mask, (1, 1, 0, 0))


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

        class RandomScale:
            def __init__(self, scaleFrom, scaleTo):
                self.scale = random.uniform(scaleFrom, scaleTo)

            def __call__(self, method, input):
                output = scale(input, method, [self.scale])
                return output[0]

        class RandomRotate:
            def __init__(self, rotateFrom, rotateTo):
                self.rotate = random.uniform(rotateFrom, rotateTo)

            def __call__(self, method, input):
                output = input.rotate(self.rotate, method)
                return output

        def getPatch():
            randomCrop = RandomCrop(trainCrop=self.trainCrop)
            if self.argument:
                randomScale = RandomScale(scaleFrom=1, scaleTo=0.5)
                # randomRotate = RandomRotate(rotateFrom=-30, rotateTo=30)

            if self.randomLR is not None:
                isLorR = random.randint(0, 1) == 1
                randomMask = list(self.mask[:])
                if self.randomLR == 'disp':
                    iStart = 2
                elif self.randomLR == 'rgb':
                    iStart = 0
                else:
                    raise Exception(f'No randomLR setting: {self.randomLR}')

                for i in range(iStart, iStart + 2):
                    randomMask[i::4] = [isLorR and original for original in randomMask[i::4]]
                    isLorR = not isLorR
            else:
                randomMask = self.mask

            def loadIm(dirsIndex, loader, scaleRatios, isRGBorDepth):
                ims = []
                if not randomMask[dirsIndex] or self.dirs[dirsIndex] is None:
                    return [np.array([], dtype=np.float32),] * len(self.loadScale)
                im0 = loader(self.dirs[dirsIndex][index])
                if type(im0) == np.ndarray:
                    im0 = Image.fromarray(im0)

                # scale first to reduce time consumption
                scaleMethod = Image.ANTIALIAS if isRGBorDepth else Image.NEAREST
                rotateMethod = Image.BICUBIC if isRGBorDepth else Image.NEAREST
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
                        pass
                    elif self.mode in ('training', 'testing', 'submission'):
                        if self.mode == 'training':
                            # random scale
                            if self.argument:
                                ims[0] = randomScale(method=scaleMethod, input=ims[0])
                                # ims[0] = randomRotate(method=rotateMethod, input=ims[0])
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
                    else:
                        raise Exception('No mode %s!' % self.mode)

                    # scale to different sizes specified by scaleRatios
                    ims += scale(ims[0], scaleMethod, multiScales)
                    ims = [transforms.ToTensor()(im) for im in ims]
                    if not isRGBorDepth and ims[0].max() == 0:
                        # print('Note: Crop has no data, recropping...')
                        return None

                ims = [np.ascontiguousarray(im, dtype=np.float32) for im, scaleRatio in zip(ims, scaleRatios)]
                if not isRGBorDepth:
                    ims = [im / self.dispScale * scaleRatio for im, scaleRatio in zip(ims, scaleRatios)]
                return ims

            gtL = loadIm(2, self.gtLoader, self.loadScale, False)
            if gtL is None:
                return None
            gtR = loadIm(3, self.gtLoader, self.loadScale, False)
            if gtR is None:
                return None

            inputL = loadIm(0, self.inputLoader, self.loadScale, True)
            inputR = loadIm(1, self.inputLoader, self.loadScale, True)

            outputs = [inputL, inputR, gtL, gtR]
            return outputs

        while True:
            outputs = getPatch()
            if outputs is not None:
                # for iIms, ims in enumerate(outputs):
                #     iCompare = iIms + 1 if iIms % 2 == 0 else iIms - 1
                #     for iScale in range(len(ims)):
                #         if ims[iScale].size == 0:
                #             ims[iScale] = np.zeros_like(outputs[iCompare][iScale])
                r = [im for scale in zip(*outputs) for im in scale]
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
