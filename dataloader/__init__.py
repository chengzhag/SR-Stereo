import torch


# cropScale: Defaultly set to loadScale to remain ratio between loaded image and cropped image.
# loadScale: A list of scale to load. Will return 4 * len(loadScale) images. Should be decreasing values.
def getDataLoader(datapath, dataset='sceneflow', trainCrop=(256, 512), batchSizes=(12, 11),
                  loadScale=(1,), mode='normal', mask=(1, 1, 1, 1), randomLR=None, subValidSet=1):
    subModes = ('subTrain', 'subEval', 'subTrainEval', 'subTest')
    if not hasattr(loadScale, '__iter__'):
        loadScale = (loadScale,)
    if dataset == 'sceneflow':
        from dataloader import listSceneFlowFiles as listFile
    elif dataset == 'kitti2012':
        if mode == 'subTest':
            from dataloader import KITTI_submission_loader2012 as listFile
        else:
            from dataloader import listKitti2012Files as listFile
    elif dataset in ('kitti2015', 'kitti2015_dense'):
        if mode == 'subTest':
            from dataloader import KITTI_submission_loader as listFile
        else:
            from dataloader import listKitti2015Files as listFile
    elif dataset == 'carla_kitti':
        from dataloader import listCarlaKittiFiles as listFile
    else:
        raise Exception('No dataloader for dataset \'%s\'!' % dataset)

    from dataloader import DataLoader as fileLoader

    paths = list(listFile.dataloader(datapath))
    pathsTrain = paths[0:4]
    if len(paths) == 8:
        pathsTest = paths[4:8]
    else:
        pathsTest = None
    if subValidSet < 1:
        pathsTest = list(zip(*pathsTest))
        pathsTest = pathsTest[:round(len(pathsTest) * subValidSet)]
        pathsTest = list(zip(*pathsTest))

    # For KITTI, images have different resolutions. Crop will be needed.
    kitti = dataset in ('kitti2012', 'kitti2015', 'kitti2015_dense')
    if dataset in ('kitti2012', 'kitti2015'):
        mask = [a and b for a, b in zip(mask, (1, 1, 1, 0))]
        dispScale = 256
        kittiScale = 1
    elif dataset == 'kitti2015_dense':
        dispScale = 170
        kittiScale = 2
    else:
        dispScale = 1
        kittiScale = 1

    if mode in subModes:
        if mode == 'subTrain':
            pathsTest = pathsTrain
        elif mode == 'subEval':
            pass
        elif mode == 'subTrainEval':
            pathsTest = [dirsTrain + dirsEval if dirsTrain is not None else None for dirsTrain, dirsEval in zip(pathsTrain, pathsTest)]
        elif mode == 'subTest':
            pathsTest = pathsTrain
        mode = 'submission'

    trainImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(*pathsTrain, trainCrop=trainCrop,
                                 kitti=kitti, loadScale=loadScale,
                                 mode=mode, mask=mask, randomLR=randomLR,
                                 dispScale=dispScale, kittiScale=kittiScale),
        batch_size=batchSizes[0], shuffle=True, num_workers=4, drop_last=False) if batchSizes[0] > 0 else None

    testImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(*pathsTest, trainCrop=trainCrop,
                                 kitti=kitti, loadScale=loadScale,
                                 mode='testing' if mode == 'training' else mode,
                                 mask=mask, randomLR=randomLR,
                                 dispScale=dispScale, kittiScale=kittiScale),
        batch_size=batchSizes[1], shuffle=False, num_workers=4, drop_last=False) if batchSizes[1] > 0 else None

    # Add dataset info to imgLoader objects
    # For KITTI, evaluation should exclude zero disparity pixels. A flag kitti will be added to imgLoader.
    for imgLoader in (trainImgLoader, testImgLoader):
        if imgLoader is not None:
            imgLoader.kitti = kitti
            imgLoader.loadScale = loadScale
            imgLoader.trainCrop = trainCrop
            imgLoader.datapath = datapath
            imgLoader.batchSizes = batchSizes

    return trainImgLoader, testImgLoader
