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

    all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = listFile.dataloader(
        datapath)

    trainImgLoader = torch.utils.data.DataLoader(
            fileLoader.myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, True, trainCrop=trainCrop),
            batch_size=batchSizes[0], shuffle=True, num_workers=8, drop_last=False) if batchSizes[0] > 0 else None

    testImgLoader = torch.utils.data.DataLoader(
            fileLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, False, trainCrop=trainCrop),
            batch_size=batchSizes[1], shuffle=False, num_workers=8, drop_last=False) if batchSizes[1] > 0 else None

    return trainImgLoader, testImgLoader
