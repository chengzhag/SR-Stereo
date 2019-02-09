import time

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

