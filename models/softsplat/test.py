from softsplatnet import Flow, Synthesis, MSFeatures
from softsplat import softsplat
import torch
import numpy as np
from PIL import Image
from flow_viz import flow_to_image
import cv2

sample_name = 'human'
torch.backends.cudnn.enabled = True
torch.set_grad_enabled(False)

flownet = Flow()
fusionnet = Synthesis()
extractor = MSFeatures()
state_dict = {
    k.replace('module', 'net'): v
    for k, v in torch.load('network-lf.pytorch').items()
}

state_dict_flow = {
    k.replace('netFlow.', ''): v
    for k, v in state_dict.items() if 'netFlow' in k
}
state_dict_fusion = {
    k.replace('netSynthesis.', ''): v
    for k, v in state_dict.items() if 'netSynthesis' in k
}
state_dict_extractor = {
    k.replace('netSynthesis.', ''): v
    for k, v in state_dict.items() if 'netSynthesis' in k
}

flownet.load_state_dict(state_dict_flow)
fusionnet.load_state_dict(state_dict_fusion)
extractor.load_state_dict(state_dict_extractor, strict=False)
flownet.cuda().eval()
fusionnet.cuda().eval()
extractor.cuda().eval()
strOne = '/home/gywu/Projects/FIFlow/demo/images/%s0.png' % sample_name
strTwo = '/home/gywu/Projects/FIFlow/demo/images/%s1.png' % sample_name
tenOne = torch.FloatTensor(
    np.ascontiguousarray(
        np.array(Image.open(strOne))[:, :, ::-1].copy().transpose(
            2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
tenTwo = torch.FloatTensor(
    np.ascontiguousarray(
        np.array(Image.open(strTwo))[:, :, ::-1].copy().transpose(
            2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

intWidth = tenOne.shape[2]
intHeight = tenOne.shape[1]

tenOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
tenTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

intPadr = (2 - (intWidth % 2)) % 2
intPadb = (2 - (intHeight % 2)) % 2

tenOne = torch.nn.functional.pad(input=tenOne,
                                 pad=[0, intPadr, 0, intPadb],
                                 mode='replicate')
tenTwo = torch.nn.functional.pad(input=tenTwo,
                                 pad=[0, intPadr, 0, intPadb],
                                 mode='replicate')

with torch.set_grad_enabled(False):
    tenStats = [tenOne, tenTwo]
    tenMean = sum([tenIn.mean([1, 2, 3], True)
                   for tenIn in tenStats]) / len(tenStats)
    tenStd = (sum([
        tenIn.std([1, 2, 3], False, True).square() +
        (tenMean - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats
    ]) / len(tenStats)).sqrt()
    tenOne = ((tenOne - tenMean) / (tenStd + 0.0000001)).detach()
    tenTwo = ((tenTwo - tenMean) / (tenStd + 0.0000001)).detach()
# end

objFlow = flownet(tenOne, tenTwo)
fflow, bflow = 0.5 * objFlow['tenForward'], 0.5 * objFlow['tenBackward']
out = fusionnet(tenOne, tenTwo, fflow, bflow, 0.5)
features = extractor(tenOne, tenTwo, fflow, bflow, 0.5)
print('INFO of Features:')
print('Length of features: ', len(features), len(features[0]))
print('Size of features: ', [features[i][0].shape for i in range(3)])
print('Input size: ', tenOne.shape)

fflow_np = fflow[0, :, :intHeight, :intWidth].cpu().numpy().transpose(1, 2, 0)

bflow_np = bflow[0, :, :intHeight, :intWidth].cpu().numpy().transpose(1, 2, 0)
fflow_rgb, bflow_rgb = [flow_to_image(x) for x in [fflow_np, bflow_np]]
Image.fromarray(fflow_rgb).save('%s_fflow.png' % (sample_name))
Image.fromarray(bflow_rgb).save('%s_bflow.png' % (sample_name))

warped_im0_ten = softsplat(tenIn=tenOne,
                           tenFlow=fflow,
                           tenMetric=None,
                           strMode='avg')
warped_im1_ten = softsplat(tenIn=tenTwo,
                           tenFlow=bflow,
                           tenMetric=None,
                           strMode='avg')
tenOne_ones = torch.ones_like(tenOne, device=tenOne.device)
warped_hole0_ten = softsplat(tenIn=tenOne_ones,
                             tenFlow=fflow,
                             tenMetric=None,
                             strMode='avg')
tenTwo_ones = torch.ones_like(tenTwo, device=tenTwo.device)
warped_hole1_ten = softsplat(tenIn=tenTwo_ones,
                             tenFlow=bflow,
                             tenMetric=None,
                             strMode='avg')

warped_im0_ten = torch.clamp(warped_im0_ten * tenStd + tenMean, 0, 1) * 255.
warped_im1_ten = torch.clamp(warped_im1_ten * tenStd + tenMean, 0, 1) * 255.
out = torch.clamp(out * tenStd + tenMean, 0, 1) * 255.

warped_im0 = (warped_im0_ten[0, :, :intHeight, :intWidth]
              ).byte().cpu().numpy().transpose(1, 2, 0)
warped_im1 = (warped_im1_ten[0, :, :intHeight, :intWidth]
              ).byte().cpu().numpy().transpose(1, 2, 0)
out = (out[0, :, :intHeight, :intWidth]).byte().cpu().numpy().transpose(
    1, 2, 0)
cv2.imwrite('%s_warped0.png' % sample_name, warped_im0)
cv2.imwrite('%s_warped1.png' % sample_name, warped_im1)
cv2.imwrite('%s_out.png' % sample_name, out)

warped_hole0_ten = torch.clamp(warped_hole0_ten, 0, 1) * 255.
warped_hole1_ten = torch.clamp(warped_hole1_ten, 0, 1) * 255.
warped_hole0 = (warped_hole0_ten[0, :, :intHeight, :intWidth]
                ).byte().cpu().numpy().transpose(1, 2, 0)
warped_hole1 = (warped_hole1_ten[0, :, :intHeight, :intWidth]
                ).byte().cpu().numpy().transpose(1, 2, 0)
cv2.imwrite('%s_hole0.png' % sample_name, warped_hole0)
cv2.imwrite('%s_hole1.png' % sample_name, warped_hole1)
