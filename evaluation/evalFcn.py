import torch

# L1 loss between gt and output
def l1(gt, output):
    if len(gt) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output - gt))  # end-point-error
    return loss


# Compute outlier proportion. Ported from kitti devkit
# http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
# 'For this benchmark, we consider a pixel to be correctly estimated if the disparity or flow end-point error is <3px or <5%'
def outlier(gt, output, npx=3, acc=0.05):
    dErr = torch.abs(gt - output)
    nTotal = torch.numel(gt)
    nWrong = torch.sum((dErr > npx) & ((dErr / gt) > acc), dtype = torch.float64)
    return nWrong / nTotal
