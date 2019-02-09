import torch
import numpy as np

def l1(gt, output):
    if len(gt) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output - gt))  # end-point-error
    return loss

# # compute n-px error, ported from kitti devkit
# def npxError(gt, output, npx=3):
#     true_disp = gt
#     index = np.argwhere(true_disp > 0)
#     gt[index[0][:], index[1][:], index[2][:]] = np.abs(
#         true_disp[index[0][:], index[1][:], index[2][:]] - output[index[0][:], index[1][:], index[2][:]])
#     correct = (gt[index[0][:], index[1][:], index[2][:]] < npx) | (
#             gt[index[0][:], index[1][:], index[2][:]] < true_disp[
#             index[0][:], index[1][:], index[2][:]] * 0.05)
#     torch.cuda.empty_cache()

# function d_err = disp_error (D_gt,D_est,tau)
#
# E = abs(D_gt-D_est);
# n_err   = length(find(D_gt>0 & E>tau(1) & E./abs(D_gt)>tau(2)));
# n_total = length(find(D_gt>0));
# d_err = n_err/n_total;

