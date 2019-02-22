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
    def __init__(self, inputLdirs, inputRdirs=None, gtLdirs=None, gtRdirs=None, status='testing',
                 trainCrop=(256, 512), kitti=False, loadScale=1, mode='normal'):
        self.inputLdirs = inputLdirs
        self.inputRdirs = inputRdirs
        self.status = status
        # in submission, only input images are needed
        self.gtLdirs = gtLdirs if self.status != 'submission' else None
        self.gtRdirs = gtRdirs if self.status != 'submission' else None
        self.inputLoader = rgbLoader
        self.gtLoader = grayLoader if kitti else pfmLoader
        self.trainCrop = trainCrop
        self.testCrop = (round(1232 * loadScale), round(368 * loadScale)) if kitti else None
        self.dispScale = 256 if kitti else 1
        self.loadScale = loadScale
        self.trainCrop = trainCrop
        self.mode = mode

    def __getitem__(self, index):
        def scale(im, method):
            w, h = im.size
            return im.resize((round(w * self.loadScale), round(h * self.loadScale)), method)

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

        def loadInput(dir, randomCrop):
            input = self.inputLoader(dir)

            if self.mode == 'raw':
                input = transforms.ToTensor()(input) if input is not None else None
                return input

            input = scale(input, Image.ANTIALIAS)

            if self.mode == 'PIL':
                pass
            elif self.mode == 'scaled':
                input = transforms.ToTensor()(input) if input is not None else None
            elif self.mode == 'normal':
                if self.status == 'training':
                    # random crop
                    input = randomCrop(input)

                elif self.status == 'testing':
                    if self.testCrop is not None:
                        # crop to the same size
                        input = testCrop(input)
                elif self.status == 'submission':
                    # do no crop
                    pass
                else:
                    raise Exception('No stats \'%s\'' % self.status)

                processed = preprocess.get_transform(augment=False)
                input = processed(input)
            else:
                raise Exception('No mode %s!' % self.mode)

            return input

        def loadGT(dir, randomCrop):
            gt = self.gtLoader(dir)
            if type(gt) == np.ndarray:
                gt = Image.fromarray(gt)

            if self.mode == 'raw':
                gt = transforms.ToTensor()(gt)
                gt = gt.squeeze()
                return gt

            gt = scale(gt, Image.NEAREST)

            if self.mode == 'PIL':
                pass
            elif self.mode == 'scaled':
                pass
            elif self.mode == 'normal':
                if self.status == 'training':
                    # random crop
                    gt = randomCrop(gt)
                elif self.status == 'testing':
                    if self.testCrop is not None:
                        # crop to the same size
                        gt = testCrop(gt)
                elif self.status == 'submission':
                    # do no crop
                    pass
                else:
                    raise Exception('No stats \'%s\'' % self.status)
            else:
                raise Exception('No mode %s!' % self.mode)

            gt = np.ascontiguousarray(gt, dtype=np.float32) / self.dispScale * self.loadScale
            return gt

        randomCrop = RandomCrop(trainCrop=self.trainCrop)

        inputL = loadInput(self.inputLdirs[index], randomCrop) if self.inputLdirs is not None else np.array([])
        inputR = loadInput(self.inputRdirs[index], randomCrop) if self.inputRdirs is not None else np.array([])

        gtL = loadGT(self.gtLdirs[index], randomCrop) if self.gtLdirs is not None else np.array([])
        gtR = loadGT(self.gtRdirs[index], randomCrop) if self.gtRdirs is not None else np.array([])

        return inputL, inputR, gtL, gtR

    def __len__(self):
        return len(self.inputLdirs)
