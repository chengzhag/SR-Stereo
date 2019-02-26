import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.pfm', 'PFM'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _scanImages(filepath, episodes, folderName):
    folderDir = os.path.join(filepath, episodes, folderName)
    imagedirs = [os.path.join(folderDir, d) for d in os.listdir(folderDir) if is_image_file(d)]
    imagedirs.sort()
    return imagedirs


def dataloader(filepath, trainProportion=0.8):
    episodes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    episodes.sort()

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_right_disp = []

    for i, episode in enumerate(episodes):
        if i + 1 <= trainProportion * len(episodes):
            all_left_img += (_scanImages(filepath, episode, 'Camera2RGB'))
            all_right_img += (_scanImages(filepath, episode, 'Camera3RGB'))
            all_left_disp += (_scanImages(filepath, episode, 'Camera2Depth'))
            all_right_disp += (_scanImages(filepath, episode, 'Camera3Depth'))
        else:
            test_left_img += (_scanImages(filepath, episode, 'Camera2RGB'))
            test_right_img += (_scanImages(filepath, episode, 'Camera3RGB'))
            test_left_disp += (_scanImages(filepath, episode, 'Camera2Depth'))
            test_right_disp += (_scanImages(filepath, episode, 'Camera3Depth'))

    return all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp


def main():
    import argparse
    from tensorboardX import SummaryWriter
    import torch
    from utils import myUtils
    import dataloader

    parser = myUtils.getBasicParser(
        ['outputFolder', 'maxdisp', 'seed', 'datapath', 'load_scale', 'nsample_save'],
        description='listCarlaKittiFiles module test')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Dataset
    trainImgLoader, _ = dataloader.getDataLoader(datapath=args.datapath, dataset='carla_kitti',
                                                 batchSizes=(1, 0),
                                                 loadScale=args.load_scale,
                                                 mode='rawScaledTensor')

    logFolder = [folder for folder in args.datapath.split('/') if folder != '']
    logFolder[-1] += '_moduleTest'
    writer = SummaryWriter(os.path.join(*logFolder))

    for iSample, sample in enumerate(trainImgLoader, 1):
        for name, im in zip(('inputL', 'inputR', 'gtL', 'gtR'), sample):
            myUtils.logFirstNIms(writer, 'listCarlaKittiFiles/' + name, im,
                                 args.maxdisp if im is not None and im.dim() == 3 else 255,
                                 global_step=iSample, n=args.nsample_save)
        if iSample >= args.nsample_save:
            break

    writer.close()


if __name__ == '__main__':
    main()
