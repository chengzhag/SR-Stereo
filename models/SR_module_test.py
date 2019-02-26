import os
import time
import torch
from utils import myUtils
from dataloader import listAllImagesInFolder as listFile
from dataloader import DataLoader as fileLoader
from models.SR import SR
import skimage
import skimage.io
import skimage.transform

def main():
    parser = myUtils.getBasicParser(
        ['outputFolder', 'datapath', 'loadmodel', 'no_cuda', 'half'],
        description='module test for class SR')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    paths = listFile.dataloader(args.datapath)
    imgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(paths, mode='rawScaledTensor'),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    # Load model
    sr = SR(cuda=args.cuda, half=args.half, stage='SR_moduleTest', dataset='testImages', saveFolderSuffix='')
    if args.loadmodel is not None:
        sr.load(args.loadmodel)

    # Predict
    saveFolder = os.path.join(sr.checkpointFolder, 'SR_module_test')
    myUtils.checkDir(saveFolder)
    tic = time.time()
    ticFull = time.time()
    for iIm, ims in enumerate(imgLoader, 1):
        name = imgLoader.dataset.name(iIm - 1)
        savePath = os.path.join(saveFolder, name)
        output = sr.predict(ims[0])
        output = output.squeeze()
        output = output.data.cpu().numpy()
        output = output.transpose(1,2,0)
        skimage.io.imsave(savePath, (output * 255).astype('uint8'))

        timeLeft = (time.time() - tic) / 60 * (len(imgLoader) - iIm)
        print('im %d/%d, %s, left %.2fmin' % (
            iIm, len(imgLoader),
            savePath, timeLeft))
        tic = time.time()
    testTime = time.time() - ticFull
    print('Full test time = %.2fmin' % (testTime / 60))

if __name__ == '__main__':
    main()