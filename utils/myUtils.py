import torch

class NameValues:
    def __init__(self, prefix, suffixes, values):
        self._pairs = []
        for suffix, value in zip(suffixes, values):
            if value is not None:
                self._pairs.append((prefix + suffix, value))

    def str(self, unit=''):
        scale = 1
        if unit == '%':
            scale = 100
        str = ''
        for name, value in self._pairs:
            str += '%s: %.2f%s, ' % (name, value * scale, unit)
        return str

    def dic(self):
        dic={}
        for name, value in self._pairs:
            dic[name] = value
        return dic

    def pairs(self):
        return self._pairs

class AutoPad:
    def __init__(self, imgs, multiple):
        self.N, self.C, self.H, self.W = imgs.size()
        self.HPad = ((self.H - 1) // multiple + 1) * multiple
        self.WPad = ((self.W - 1) // multiple + 1) * multiple

    def pad(self, imgs, cuda):
        imgsPad = torch.zeros([self.N, self.C, self.HPad, self.WPad], dtype=imgs.dtype, device='cuda' if cuda else 'cpu')
        imgsPad[:, :, (self.HPad - self.H):, (self.WPad - self.W):] = imgs
        return imgsPad

    def unpad(self, imgs):
        imgs = imgs[:, (self.HPad - self.H):, (self.WPad - self.W):]
        return imgs

# Flip among W dimension. For NCHW data type.
def flipLR(im):
    return im.flip(-1)

def assertDisp(dispL=None, dispR=None):
    if (dispL is None or dispL.numel() == 0) and (dispR is None or dispR.numel() == 0):
        raise Exception('No disp input!')

# Log First n disparity maps into tensorboard
# Log All disparity maps if n == 0
def logFirstNdis(writer, name, disp, maxdisp, global_step=None, n=0):
    if disp is not None and n > 0:
        n = min(n, disp.size(0))
        disp = disp[:n, :, :]
        disp[disp > maxdisp] = maxdisp
        disp = disp / maxdisp
        disp = gray2rgb(disp)
        writer.add_images(name, disp, global_step=global_step)

def gray2rgb(im):
    return im.unsqueeze(1).repeat(1, 3, 1, 1)
