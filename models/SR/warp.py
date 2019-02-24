import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gc


def warp(left, right, displ, dispr):
    if displ.dim() == 3: displ = displ.unsqueeze(0)
    if dispr.dim() == 3: dispr = dispr.unsqueeze(0)

    # pdb.set_trace()
    b, c, h, w = left.size()
    y0, x0 = np.mgrid[0:h, 0:w]
    y = np.expand_dims(y0, 0)
    y = np.expand_dims(y, 0).repeat(b, 0)
    x = np.expand_dims(x0, 0)
    x = np.expand_dims(x, 0).repeat(b, 0)
    # print(x.shape,y.shape)
    grid = np.concatenate((x, y), 1)

    if displ.is_cuda:
        grid = torch.from_numpy(grid).type_as(displ).cuda()
        y_zeros = torch.zeros(displ.size()).type_as(displ).cuda()
    else:
        grid = torch.from_numpy(grid)
        y_zeros = torch.zeros(displ.size())
    flol = torch.cat((displ, y_zeros), 1)
    flor = torch.cat((dispr, y_zeros), 1)
    gridl = grid - flol
    gridr = grid + flor

    gridl[:, 0, :, :] = 2.0 * gridl[:, 0, :, :] / max(w - 1, 1) - 1.0
    gridl[:, 1, :, :] = 2.0 * gridl[:, 1, :, :] / max(h - 1, 1) - 1.0
    gridr[:, 0, :, :] = 2.0 * gridr[:, 0, :, :] / max(w - 1, 1) - 1.0
    gridr[:, 1, :, :] = 2.0 * gridr[:, 1, :, :] / max(h - 1, 1) - 1.0
    vgridl = Variable(gridl)
    vgridr = Variable(gridr)

    vgridl = vgridl.permute(0, 2, 3, 1)
    vgridr = vgridr.permute(0, 2, 3, 1)

    Drwarp2l = nn.functional.grid_sample(dispr, vgridl)
    Dlwarp2r = nn.functional.grid_sample(displ, vgridr)

    locl = abs(displ - Drwarp2l)
    rocl = abs(dispr - Dlwarp2r)

    th = 0.5
    rocl[rocl <= th] = th
    rocl[rocl > th] = 0
    rocl[rocl > 0] = 1
    locl[locl <= th] = th
    locl[locl > th] = 0
    locl[locl > 0] = 1

    Irwarp2l = nn.functional.grid_sample(right, vgridl)
    Ilwarp2r = nn.functional.grid_sample(left, vgridr)
    if displ.is_cuda:
        maskl_ = torch.autograd.Variable(torch.ones(displ.size())).type_as(displ).cuda()
        maskr_ = torch.autograd.Variable(torch.ones(displ.size())).type_as(displ).cuda()
    else:
        maskl_ = torch.autograd.Variable(torch.ones(displ.size()))
        maskr_ = torch.autograd.Variable(torch.ones(displ.size()))
    maskl_ = nn.functional.grid_sample(maskl_, vgridl)
    maskr_ = nn.functional.grid_sample(maskr_, vgridr)
    maskl_[maskl_ < 0.999] = 0
    maskl_[maskl_ > 0] = 1
    maskr_[maskr_ < 0.999] = 0
    maskr_[maskr_ > 0] = 1
    imglw = Irwarp2l * maskl_ * locl
    imgrw = Ilwarp2r * maskr_ * rocl
    maskl = imglw.sum(dim=1, keepdim=True)
    maskr = imgrw.sum(dim=1, keepdim=True)
    maskl[maskl < 0.999] = 0
    maskl[maskl > 0] = 1
    maskr[maskr < 0.999] = 0
    maskr[maskr > 0] = 1

    # outimgl = torch.cat([left, imglw, maskl], 1)
    # outimgr = torch.cat([right, imgrw, maskr], 1)

    for x in list(locals()):
        del locals()[x]
    gc.collect()
    return imglw, imgrw, maskl, maskr


def main():
    from utils import myUtils
    from tensorboardX import SummaryWriter
    import os
    from evaluation import evalFcn
    import dataloader
    parser = myUtils.getBasicParser(['maxdisp', 'datapath', 'no_cuda', 'seed', 'eval_fcn',
                                     'dataset', 'load_scale', 'nsample_save'],
                                    description='warp module test')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    _, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                batchSizes=(0, 1),
                                                loadScale=args.load_scale,
                                                mode='rawScaledTensor')

    logFolder = [folder for folder in args.datapath.split('/') if folder != '']
    logFolder[-1] += '_moduleTest'
    writer = SummaryWriter(os.path.join(*logFolder))

    for iSample, sample in enumerate(testImgLoader, 1):
        if args.cuda:
            sample = [s.cuda() for s in sample]
        imglw, imgrw, maskl, maskr = warp(*sample)

        masklRGB = maskl.byte().repeat(1, 3, 1, 1)
        maskrRGB = maskr.byte().repeat(1, 3, 1, 1)
        errorL = getattr(evalFcn, args.eval_fcn)(sample[0][masklRGB], imglw[masklRGB])
        errorR = getattr(evalFcn, args.eval_fcn)(sample[1][maskrRGB], imgrw[maskrRGB])

        for name, value in myUtils.NameValues(('L', 'R'), (errorL, errorR), prefix='error').pairs():
            writer.add_scalar('warp/' + name, value, iSample)
        for name, im, range in zip(
                ('inputL', 'inputR', 'gtL', 'gtR', 'warpL', 'warpR', 'disocclusionsMaskL', 'disocclusionsMaskR'),
                sample + [imglw, imgrw, maskl, maskr],
                (255, 255, args.maxdisp, args.maxdisp, 255, 255, 1, 1)
        ):
            myUtils.logFirstNIms(writer, 'warp/' + name, im, range,
                                 global_step=iSample, n=args.nsample_save)

        if iSample >= args.nsample_save:
            break

    writer.close()


if __name__ == '__main__':
    main()
