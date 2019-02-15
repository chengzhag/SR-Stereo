import torch

# cropScale: defaultly set to loadScale to remain ratio between loaded image and cropped image
def getDataLoader(datapath, dataset='sceneflow', trainCrop=(512, 256), batchSizes=(12, 11), loadScale=1, cropScale=None):
    if dataset == 'sceneflow':
        from dataloader import listSceneFlowFiles as listFile
    elif dataset == 'kitti2012':
        from dataloader import listKitti2012Files as listFile
    elif dataset == 'kitti2015':
        from dataloader import listKitti2015Files as listFile
    elif dataset == 'carla_kitti':
        from dataloader import listCarlaKittiFiles as listFile
    else:
        raise Exception('No dataloader for dataset \'%s\'!' % dataset)

    from dataloader import DataLoader as fileLoader

    paths = listFile.dataloader(datapath)
    pathsTrain = paths[0:4]
    pathsTest = paths[4:8]

    # For KITTI, images have different resolutions. Crop will be needed.
    kitti = dataset in ('kitti2012', 'kitti2015')

    cropScale = loadScale if cropScale is None else cropScale
    trainImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(*pathsTrain, training=True, trainCrop=trainCrop,
                                 kitti=kitti, loadScale=loadScale ,cropScale=cropScale),
        batch_size=batchSizes[0], shuffle=True, num_workers=8, drop_last=False) if batchSizes[0] > 0 else None

    testImgLoader = torch.utils.data.DataLoader(
        fileLoader.myImageFloder(*pathsTest, training=False, trainCrop=trainCrop,
                                 kitti=kitti, loadScale=loadScale, cropScale=cropScale),
        batch_size=batchSizes[1], shuffle=False, num_workers=8, drop_last=False) if batchSizes[1] > 0 else None

    # For KITTI, evaluation should exclude zero disparity pixels. A flag kitti will be added to imgLoader.
    if trainImgLoader is not None:
        trainImgLoader.kitti = kitti
        trainImgLoader.loadScale = loadScale
        trainImgLoader.cropScale = cropScale
    if testImgLoader is not None:
        testImgLoader.kitti = kitti
        testImgLoader.loadScale = loadScale
        testImgLoader.cropScale = cropScale

    return trainImgLoader, testImgLoader
