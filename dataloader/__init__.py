import torch


def getDataLoader(datapath, dataset='sceneflow', trainCrop=(256, 512), batchSizes=(12, 11)):
    if dataset == 'sceneflow':
        from dataloader import listSceneFlowFiles as listFile
        from dataloader import SceneFlowLoader as fileLoader
    elif dataset == 'kitti2012':
        from dataloader import listKitti2012Files as listFile
        from dataloader import KittiLoader as fileLoader
    elif dataset == 'kitti2015':
        from dataloader import listKitti2015Files as listFile
        from dataloader import KittiLoader as fileLoader
    elif dataset == 'carla_kitti':
        from dataloader import listCarlaKittiFiles as listFile
        from dataloader import KittiLoader as fileLoader
    else:
        raise Exception('No dataloader for dataset \'%s\'!' % dataset)

    paths = listFile.dataloader(datapath)
    if dataset == 'sceneflow' or dataset == 'carla_kitti':
        # Datasets which have both disparity maps
        pathsTrain = paths[0:4]
        pathsTest = paths[4:8]
    else:
        # Datasets which only have left disparity maps
        pathsTrain = paths[0:3]
        pathsTest = paths[3:6]

    # For KITTI, images have different resolutions. Crop will be needed.
    if dataset == 'kitti2012' or dataset == 'kitti2015':
        testCrop = True
    else:
        testCrop = False

    trainImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(*pathsTrain, training=True, trainCrop=trainCrop, testCrop=testCrop),
        batch_size=batchSizes[0], shuffle=True, num_workers=8, drop_last=False) if batchSizes[0] > 0 else None

    testImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(*pathsTest, training=False, trainCrop=trainCrop, testCrop=testCrop),
        batch_size=batchSizes[1], shuffle=False, num_workers=8, drop_last=False) if batchSizes[1] > 0 else None

    # For KITTI, evaluation should exclude zero disparity pixels. A flag kitti will be added to imgLoader.
    def setKitti(kitti):
        if trainImgLoader is not None:
            trainImgLoader.kitti = kitti
        if testImgLoader is not None:
            testImgLoader.kitti = kitti
    if dataset == 'kitti2012' or dataset == 'kitti2015':
        setKitti(True)
    else:
        setKitti(False)
        
    return trainImgLoader, testImgLoader
