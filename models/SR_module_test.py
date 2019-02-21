import os
import time
import torch
from utils import myUtils
from dataloader import listAllImagesInFolder as listFile
from models.SR import SR

def main():
    parser = myUtils.getBasicParser(
        ['datapath', 'loadmodel', 'no_cuda'],
        description='module test for class SR')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Dataset
    paths = listFile.dataloader(args.datapath)

    # Load model
    sr = SR(cuda=args.cuda, stage='SR_moduleTest', dataset='testImage', saveFolderSuffix='')
    sr.load(args.loadmodel)



if __name__ == '__main__':
    main()