import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gc


def warp(left, right, displ, dispr):
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

    outimgl = torch.cat([left, imglw, maskl], 1)
    outimgr = torch.cat([right, imgrw, maskr], 1)

    '''
    make_dir('./results')
    imglw=torch.squeeze(imglw,0).permute(1,2,0)
    imgrw=torch.squeeze(imgrw,0).permute(1,2,0)
    imgL0=torch.squeeze(imgfusionl,0).permute(1,2,0)
    imgR0=torch.squeeze(imgfusionr,0).permute(1,2,0)
    imglw=imglw.cpu().detach().numpy().astype('uint8')
    imgrw=imgrw.cpu().detach().numpy().astype('uint8')
    imgL0=imgL0.cpu().detach().numpy().astype('uint8')
    imgR0=imgR0.cpu().detach().numpy().astype('uint8')
    #make_dir('./results')
    skimage.io.imsave('./results/'+namel[0].split('/')[-4]+namel[0].split('/')[-3]+"lw"+namel[0].split('/')[-1],imglw)
    skimage.io.imsave('./results/'+namel[0].split('/')[-4]+namel[0].split('/')[-3]+"rw"+namel[0].split('/')[-1],imgrw)
    skimage.io.imsave('./results/'+namel[0].split('/')[-4]+namel[0].split('/')[-3]+"lf"+namel[0].split('/')[-1],imgL0)
    skimage.io.imsave('./results/'+namel[0].split('/')[-4]+namel[0].split('/')[-3]+"rf"+namel[0].split('/')[-1],imgR0)
    '''
    for x in list(locals()):
        del locals()[x]
    gc.collect()
    return outimgl, outimgr, imglw, imgrw


def main():
    from utils import myUtils
    from tensorboardX import SummaryWriter
    import os
    import argparse
    parser = argparse.ArgumentParser(description='warp')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--datapath', default='../datasets/sceneflow/',
                        help='datapath')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval_fcn', type=str, default='l1',
                        help='evaluation function used in testing')
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        help='evaluation function used in testing')
    parser.add_argument('--load_scale', type=float, default=1,
                        help='scaling applied to data during loading')
    parser.add_argument('--crop_scale', type=float, default=None,
                        help='scaling applied to data during croping')
    parser.add_argument('--nsample_save', type=int, default=5,
                        help='save n samples as png images')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataset
    import dataloader
    _, testImgLoader = dataloader.getDataLoader(datapath=args.datapath, dataset=args.dataset,
                                                 batchSizes=(0, 1),
                                                 loadScale=args.load_scale, cropScale=args.crop_scale, mode='raw')

    logFolder = [folder for folder in args.datapath.split('/') if folder != '']
    logFolder[-1] += '_warpTest'
    writer = SummaryWriter(os.path.join(*logFolder))

    for iSample, sample in enumerate(testImgLoader, 1):
        if args.cuda:
            sample = [s.cuda() for s in sample]
        _, _, imglw, imgrw = warp(*sample)

        for name, im  in zip(('inputL', 'inputR', 'gtL', 'gtR', 'warpL', 'warpR'), sample + [imglw, imgrw]):
            myUtils.logFirstNdis(writer, 'moduleTesting/' + name, im,
                                 args.maxdisp if im is not None and im.dim() == 3 else 255,
                                 global_step=iSample, n=args.nsample_save)

        if iSample >= args.nsample_save:
            break

    writer.close()


if __name__ == '__main__':
    main()
