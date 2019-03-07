from .PSMNet import PSMNet
from .PSMNetDown import PSMNetDown
from .SRStereo import SRStereo
from .SRdispStereo import SRdispStereo

# Abandoned
# class PSMNet_TieCheng(Stereo):
#     # dataset: only used for suffix of saveFolderName
#     def __init__(self, maxdisp=192, dispScale=1, cuda=True, half=False, stage='unnamed', dataset=None,
#                  saveFolderSuffix=''):
#         super(PSMNet_TieCheng, self).__init__(maxdisp, dispScale, cuda, half, stage, dataset, saveFolderSuffix)
#         self.getModel = rawPSMNet_TieCheng
#
#     def predict(self, batch, mask=(1, 1)):
#         self.predictPrepare()
#         inputs = batch.lowestResRGBs()
#
#         with torch.no_grad():
#             imgL, imgR = autoPad.pad(inputs)
#             outputs = self.model(imgL, imgR)
#             outputs = autoPad.unpad(outputs)
#             return tuple(outputs)
