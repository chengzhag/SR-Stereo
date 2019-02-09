import argparse
import time
import torch
from dataloader import listSceneFlowFile
from dataloader import SceneFlowLoader
from models import Stereo

def test(stereo, testImgLoader, mode='both', type='outlier', kitti=False):
    if mode == 'both':
        totalTestScores = [0, 0, 0]
        tic = time.time()
        for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(testImgLoader):
            if stereo.cuda:
                imgL, imgR = imgL.cuda(), imgR.cuda()
            scoreAvg, [scoreL, scoreR] = stereo.test(imgL, imgR, dispL, dispR, type=type, kitti=kitti)
            totalTestScores = [total + batch for total, batch in zip(totalTestScores, [scoreAvg, scoreL, scoreR])]
            timeLeft = (time.time() - tic) / 3600 * (len(testImgLoader) - batch_idx - 1)

            scoresPrint = [scoreAvg, scoreL, scoreR] + [l / (batch_idx + 1) for l in totalTestScores]
            if type == 'outlier':
                print(
                    'it %d/%d, scoreAvg %.2f%%, scoreL %.2f%%, scoreR %.2f%%, scoreTotal %.2f%%, scoreLTotal %.2f%%, scoreRTotal %.2f%%, left %.2fh' % tuple(
                        [batch_idx, len(testImgLoader)] + [s * 100 for s in scoresPrint] + [timeLeft]))
            else:
                print(
                    'it %d/%d, lossAvg %.2f, lossL %.2f, lossR %.2f, lossTotal %.2f, lossLTotal %.2f, lossRTotal %.2f, left %.2fh' % tuple(
                        [batch_idx, len(testImgLoader)] + scoresPrint + [timeLeft]))
            tic = time.time()
        return [l / len(testImgLoader) for l in totalTestScores]
    elif mode == 'left' or mode == 'right':
        totalTestScore = 0
        tic = time.time()
        for batch_idx, (imgL, imgR, dispGT) in enumerate(testImgLoader):
            if stereo.cuda:
                imgL, imgR = imgL.cuda(), imgR.cuda()
            if mode == 'left':
                score = stereo.test(imgL, imgR, dispL=dispGT, type=type, kitti=kitti)
            else:
                score = stereo.test(imgL, imgR, dispR=dispGT, type=type, kitti=kitti)
            totalTestScore += score
            timeLeft = (time.time() - tic) / 3600 * (len(testImgLoader) - batch_idx - 1)

            scoresPrint = [score, totalTestScore/(batch_idx + 1)]
            if type == 'outlier':
                print(
                    'it %d/%d, score %.2f%%, totalTestScore %.2f%%, left %.2fh' % tuple(
                        [batch_idx, len(testImgLoader)] + [s * 100 for s in scoresPrint] + [timeLeft]))
            else:
                print(
                    'it %d/%d, loss %.2f, totalTestLoss %.2f, left %.2fh' % tuple(
                        [batch_idx, len(testImgLoader)] + scoresPrint + [timeLeft]))
            tic = time.time()
        return totalTestScore/ len(testImgLoader)

def main():
    parser = argparse.ArgumentParser(description='Stereo')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='PSMNet',
                        help='select model')
    parser.add_argument('--datapath', default='../datasets/sceneflow/',
                        help='datapath')
    parser.add_argument('--loadmodel', default='logs/pretrained/PSMNet_pretrained_sceneflow.tar',
                        help='load model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test_type', type=str, default='outlier',
                        help='evaluation type used in testing')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # TODO: Add code to test given model and different dataset
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

        _, _, _, _, test_left_img, test_right_img, test_left_disp, test_right_disp = listSceneFlowFile.dataloader(
        args.datapath)

    testImgLoader = torch.utils.data.DataLoader(
        SceneFlowLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, False),
        batch_size=11, shuffle=False, num_workers=8, drop_last=False)

    stereo = getattr(Stereo, args.model)(maxdisp=args.maxdisp, cuda=args.cuda)
    stereo.load(args.loadmodel)

    # TEST
    totalTestScores = test(stereo=stereo, testImgLoader=testImgLoader, mode='both', type=args.test_type)

    # SAVE test information
    saveDir = args.savemodel + 'testinformation.tar'
    torch.save({
        'scoreAvg': totalTestScores[0],
        'scoreL': totalTestScores[1],
        'scoreR': totalTestScores[2]
    }, saveDir)

if __name__ == '__main__':
    main()

