from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import numpy as np
from scipy import interpolate
import cv2
from . import common
import pdb
import gc
class EDSR(nn.Module):
    def __init__(self,scale, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.scale=scale
        n_resblock = 16
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        self.rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = common.MeanShift(255,rgb_std)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        #m_tail_1 = [
        #    common.Upsampler_part1(conv, scale, n_feats, act=False),
        #]
        m_tail = [
            common.Upsampler(conv, self.scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, 3, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        #self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        #self.tail1 = nn.Sequential(*m_tail_1)
        self.tail = nn.Sequential(*m_tail)
    def forward(self,img):
        #pdb.set_trace()
        b,c,h,w=img.size()
        reshape_img=torch.reshape(img,(b,c,h*w))/255.0#(1,3,32768)
        #print(reshape_img.size())
        mean=torch.squeeze(torch.mean(reshape_img,(2)))#e.g. [0.4,0.38,0.41]

        img_=common.MeanShift(255,mean,self.rgb_std,-1)(img)

        
        '''
	    img_reshape0=torch.reshape(img,(b,c,self.scale,h//self.scale,self.scale,w//self.scale)).permute(0,1,3,5,2,4)
        img_reshape1=torch.reshape(img_reshape0,(b,c,h//self.scale,w//self.scale,self.scale*self.scale)).permute(0,1,4,2,3)
        smallimg0=img_reshape1[:,:,0,:,:]
        img_reshape2=torch.reshape(img_reshape1,(b,c*self.scale*self.scale,h//self.scale,w//self.scale))
        '''
        x = self.head(img_)
        res = self.body(x)
        res += x
        x = self.tail(res)
        #x = self.tail2(x_l)
        x =common.MeanShift(255,mean,self.rgb_std,1)(x)
        
        return x,res
def warp(left,right,displ,dispr):
	#pdb.set_trace()
	b,c,h,w=left.size()
	y0,x0=np.mgrid[0:h,0:w]
	y = np.expand_dims(y0, 0)
	y = np.expand_dims(y, 0).repeat(b,0)
	x = np.expand_dims(x0, 0)
	x = np.expand_dims(x, 0).repeat(b,0)
	#print(x.shape,y.shape)
	grid=np.concatenate((x,y),1)

	if cuda:
		grid=torch.from_numpy(grid).cuda(2).float()
		y_zeros=torch.zeros(displ.size()).cuda(2)
	else:
		grid=torch.from_numpy(grid).float()
		y_zeros=torch.zeros(displ.size())          
	flol=torch.cat((displ,y_zeros),1).float()
	flor=torch.cat((dispr,y_zeros),1).float()
	gridl=grid-flol
	gridr=grid+flor

	gridl[:, 0, :, :] = 2.0 * gridl[:, 0, :, :] / max(w - 1, 1) - 1.0
	gridl[:, 1, :, :] = 2.0 * gridl[:, 1, :, :] / max(h - 1, 1) - 1.0
	gridr[:, 0, :, :] = 2.0 * gridr[:, 0, :, :] / max(w - 1, 1) - 1.0
	gridr[:, 1, :, :] = 2.0 * gridr[:, 1, :, :] / max(h - 1, 1) - 1.0
	vgridl = Variable(gridl)
	vgridr = Variable(gridr)

	vgridl = vgridl.permute(0, 2, 3, 1)
	vgridr = vgridr.permute(0, 2, 3, 1)

	Drwarp2l=nn.functional.grid_sample(dispr,vgridl)
	Dlwarp2r=nn.functional.grid_sample(displ,vgridr)

	locl=abs(displ-Drwarp2l)
	rocl=abs(dispr-Dlwarp2r)

	th=0.5
	rocl[rocl<= th]=th
	rocl[rocl > th] = 0
	rocl[rocl > 0]=1
	locl[locl <= th]=th
	locl[locl > th]=0
	locl[locl > 0]=1   

	Irwarp2l=nn.functional.grid_sample(right,vgridl)
	Ilwarp2r=nn.functional.grid_sample(left,vgridr)
	if cuda:
		maskl_ = torch.autograd.Variable(torch.ones(displ.size())).cuda(2)
		maskr_ = torch.autograd.Variable(torch.ones(displ.size())).cuda(2)
	else:
		maskl_ = torch.autograd.Variable(torch.ones(displ.size()))
		maskr_ = torch.autograd.Variable(torch.ones(displ.size()))
	maskl_ = nn.functional.grid_sample(maskl_, vgridl)
	maskr_ = nn.functional.grid_sample(maskr_, vgridr)
	maskl_[maskl_ < 0.999] = 0
	maskl_[maskl_ > 0] = 1
	maskr_[maskr_ < 0.999] = 0
	maskr_[maskr_ > 0] = 1
	imglw=Irwarp2l*maskl_*locl
	imgrw=Ilwarp2r*maskr_*rocl
	maskl=imglw.sum(dim=1,keepdim=True)
	maskr=imgrw.sum(dim=1,keepdim=True)
	maskl[maskl < 0.999] = 0
	maskl[maskl > 0] = 1
	maskr[maskr < 0.999] = 0
	maskr[maskr > 0] = 1

	outimgl=torch.cat([left,imglw,maskl],1)
	outimgr=torch.cat([right,imgrw,maskr],1)

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
	for x in locals().keys():
		del locals()[x]
	gc.collect()
	return outimgl,outimgr,imglw,imgrw
class SRD(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(SRD, self).__init__()
        n_resblock = 12
        n_feats = 24
        kernel_size = 3
        act = nn.ReLU(True)
        self.rgb_std = (1.0, 1.0, 1.0)
        m_head=[conv(7,n_feats,kernel_size)]
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblock)
        ]  
        m_tail=[conv(n_feats, 3, kernel_size)]		
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.stackimgs=nn.Sequential(nn.PixelShuffle(2))
        fu=[nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1,bias=True)]
        fu=fu+[common.ResBlock(
                conv, 16, 3, act=act, res_scale=0.2) for _ in range(8)]
        self.fusion = nn.Sequential(*fu)
        self.lastconv=nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True))
        self.lowfeature = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True))
        self.attention=nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True))
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        '''

    def forward(self,left,right,displ,dispr,srlf,srrf):
        displ_=displ.unsqueeze(0)
        dispr_=dispr.unsqueeze(0)
        srdl=self.stackimgs(displ_*2)
        srdr=self.stackimgs(dispr_*2)
        dl_=self.lowfeature(srdl/2.0)
        dr_=self.lowfeature(srdr/2.0)
        #dlfeat=torch.cat([displ_,pl.repeat(1,4,1,1)],1)
        #drfeat=torch.cat([dispr_,pr.repeat(1,4,1,1)],1)
        dlfeat=torch.cat([displ_,dl_],1)
        drfeat=torch.cat([dispr_,dr_],1)
        #pdb.set_trace()
        dl=self.fusion(dlfeat)
        dr=self.fusion(drfeat)
        atl=torch.nn.functional.softmax(self.attention(srlf),1)
        atr=torch.nn.functional.softmax(self.attention(srrf),1)
        #pdb.set_trace()
        dl=dl*atl
        dr=dr*atr
        dl=self.lastconv(dl)
        dr=self.lastconv(dr)
        
        srwl,srwr,imglw,imgrw=warp(left,right,srdl,srdr)	
        srwl = self.head(srwl)
        srwr = self.head(srwr)
        resl = self.body(srwl)
        resr = self.body(srwr)
        resl += srwl
        resr += srwr
        srwl = self.tail(resl)
        srwr = self.tail(resr)
        		
        return srwl,srwr,dl,dr,srdl,srdr

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self,left,right):
        #pdb.set_trace()
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)


        #matching
        costl = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        costr = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
                costl[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                costl[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
                costr[:, :refimg_fea.size()[1], i, :,:-i]   = targetimg_fea[:,:,:,:-i]
                costr[:, refimg_fea.size()[1]:, i, :,:-i] = refimg_fea[:,:,:,i:]

            else:
                costl[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                costl[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
                costr[:, :refimg_fea.size()[1], i, :,:]   = targetimg_fea
                costr[:, refimg_fea.size()[1]:, i, :,:]   = refimg_fea
        costl = costl.contiguous()
        costr = costr.contiguous()
        costs=[costl, costr]
        pred1s=[]
        pred2s=[]
        pred3s=[]
        for i in range(2):
            cost=costs[i]
            cost0 = self.dres0(cost)
            cost0 = self.dres1(cost0) + cost0

            out1, pre1, post1 = self.dres2(cost0, None, None) 
            out1 = out1+cost0

            out2, pre2, post2 = self.dres3(out1, pre1, post1) 
            out2 = out2+cost0

            out3, pre3, post3 = self.dres4(out2, pre1, post2) 
            out3 = out3+cost0

            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2) + cost1
            cost3 = self.classif3(out3) + cost2

            if self.training:
                cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
                cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

                cost1 = torch.squeeze(cost1,1)
                pred1 = F.softmax(cost1,dim=1)
                pred1 = disparityregression(self.maxdisp)(pred1)

                cost2 = torch.squeeze(cost2,1)
                pred2 = F.softmax(cost2,dim=1)
                pred2 = disparityregression(self.maxdisp)(pred2)
                pred1s.append(pred1)
                pred2s.append(pred2)
                cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
                cost3 = torch.squeeze(cost3,1)
                pred3 = F.softmax(cost3,dim=1)
                pred3 = disparityregression(self.maxdisp)(pred3)
                pred3s.append(pred3)
            else:
                cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
                cost3 = torch.squeeze(cost3,1)
                pred3 = F.softmax(cost3,dim=1)
                pred3 = disparityregression(self.maxdisp)(pred3)
                pred3s.append(pred3)
        #srdispl=self.stackimgs(pred3s[0].unsqueeze(0))
        #srdispr=self.stackimgs(pred3s[1].unsqueeze(0))
        #pl=self.fusion(pred3s[0])
        #pr=self.fusion(pred3s[1])

        if self.training:
            return pred1s[0], pred2s[0], pred3s[0],pred1s[1], pred2s[1], pred3s[1]
        else:
            return pred3s[0],pred3s[1]
