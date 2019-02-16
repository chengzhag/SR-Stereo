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
    imagedirs =  [os.path.join(folderDir, d) for d in os.listdir(folderDir) if is_image_file(d)]
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
    import random
    from dataloader.DataLoader import myImageFloder
    from torchvision import transforms
    from PIL import Image
    from utils import myUtils

    parser = argparse.ArgumentParser(description='CarlaKitti')
    parser.add_argument('--filepath', type=str, default='../datasets/carla_kitti/carla_kitti_sr_highquality',
                        help='filepath to load')
    parser.add_argument('--savepath', type=str, default='../datasets/carla_kitti/carla_kitti_sr_highquality_png',
                        help='filepath to save')
    parser.add_argument('--nsample_save', type=int, default=1,
                        help='save n samples as png images')
    args = parser.parse_args()
    
    all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = dataloader(
        args.filepath)


    for train, test in zip((all_left_img, all_right_img, all_left_disp, all_right_disp), (test_left_img, test_right_img, test_left_disp, test_right_disp)):
        train += test

    allImgLoader = myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, training=False, kitti=False, raw=True)

    folderDirs = []
    for path in (all_left_img, all_right_img, all_left_disp, all_right_disp):
        folderDir, _ = os.path.split(path[0])
        folderDirs.append(os.path.join(args.savepath, folderDir.split('/')[-1]))
        myUtils.checkDir(folderDirs[-1])

    for isample, index in enumerate(random.sample(range(len(all_left_img)), args.nsample_save)):
        ims = allImgLoader[index]
        for im, folder in zip(ims, folderDirs):
            im.save(os.path.join(folder, '%05d.tiff' % isample))

if __name__ == '__main__':
    main()

