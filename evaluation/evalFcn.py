import torch
import numpy as np

def getEvalFcn(type):
    if '_' in type:
        params = type.split('_')
        type = params[-1]
        params = [float(param) for param in params[:-1]]
        return lambda gt, output: globals()[type](gt, output, *params)
    else:
        return globals()[type]

# L1 loss between gt and output
def l1(gt, output):
    if len(gt) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output - gt)).item()  # end-point-error
    return loss


# Compute outlier proportion. Ported from kitti devkit
# http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
# 'For this benchmark, we consider a pixel to be correctly estimated if the disparity or flow end-point error is <3px or <5%'
def outlier(gt, output, npx=3, acc=0.05):
    dErr = torch.abs(gt - output)
    nTotal = float(torch.numel(gt))
    nWrong = float(torch.sum((dErr > npx) & ((dErr / gt) > acc)).item())
    return nWrong / nTotal * 100

def outlierPSMNet(disp_true, pred_disp):
    disp_true = disp_true.squeeze(1)
    pred_disp = pred_disp.squeeze(1)
    disp_true = disp_true.data.cpu()
    pred_disp = pred_disp.data.cpu()
    # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
                disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
            index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return (1 - (float(torch.sum(correct)) / float(len(index[0])))) * 100