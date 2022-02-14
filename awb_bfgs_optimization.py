from matplotlib import pyplot as plt
import numpy as np
import csv
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.optimize import minimize, rosen, rosen_der

import cv2

visualize = True

def vis_img(image, visualize, cmap=None):
    if visualize:
        try:
            if cmap is not None:
                plt.imshow(image, cmap=cmap)
            else:
                plt.imshow(image)
            plt.show()
        except:
            pass


def PSNR(original, processed):
    ''' Peak signal-to-noise ratio (PSNR) is the ratio between the maximum possible power
        of an image and the power of corrupting noise that affects the quality of its representation. '''
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) # dB
    return psnr

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error

def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def standard_deviation_weighted_grey_world(nimg, params):
    gain_r, gain_g, gain_b = params
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.minimum(nimg[0]*gain_r,255)
    nimg[1] = np.minimum(nimg[1]*gain_g,255)
    nimg[2] = np.minimum(nimg[2]*gain_b,255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


# Optimization : raw image -> ISP -> processed image -> loss -> optimization -> ...

def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    # nimg = np.asarray(pimg)
    nimg = np.array(pimg).astype(np.uint8)
    # nimg.flags.writeable = True
    # nimg.setflags(write=1)
    return nimg

def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

raw_path = './raw/source02.png'
groundtruth_path = './raw/target02.png'

rawimg = Image.open(raw_path)
# rawimg = rawimg.resize((332, 467))
# rawimg.show()
imgsz = rawimg.size

groundtruth = Image.open(groundtruth_path)
groundtruth = groundtruth.resize(imgsz)
# groundtruth.show()

init_params = [0.5, 0.5, 0.5]
# opt_params = [0.50000073, 0.50000114, 0.50000132]

pimg_gw = to_pil(grey_world(from_pil(rawimg)))
# pimg_gw.show()
pimg_sdwgw = to_pil(standard_deviation_weighted_grey_world(from_pil(rawimg), init_params))
# pimg_sdwgw.show()
# pimg_sdwgw02 = to_pil(standard_deviation_weighted_grey_world(from_pil(rawimg), opt_params))


mse_gw = mse(from_pil(pimg_gw), from_pil(groundtruth))
ssim_gw = ssim(from_pil(pimg_gw), from_pil(groundtruth), multichannel=True)
print('grey world - mse : %f, ssim : %f'%(mse_gw, ssim_gw))

mse_sdwgw = mse(from_pil(pimg_sdwgw), from_pil(groundtruth))
ssim_sdwgw = ssim(from_pil(pimg_sdwgw), from_pil(groundtruth), multichannel=True)
print('init sdwgw - mse : %f, ssim : %f'%(mse_sdwgw, ssim_sdwgw))
# print(cv2.PNSR(from_pil(pimg_gw), from_pil(groundtruth)))


#
source = from_pil(rawimg)
target = from_pil(groundtruth)

def func(parameter):
    processed = standard_deviation_weighted_grey_world(source, parameter)
    loss = ssim(processed, target, multichannel=True)
    # loss = mse(processed, target)
    return loss
#

fitted_params = [0.5, 0.5, 0.5]
# for i in range(10):
#     print('\n --- %d iter ---- \n'%i)
result = minimize(func, fitted_params, method='BFGS', jac=rosen_der)
fitted_params = result.x
print('loss : ', result.fun, ' params : ', fitted_params)

pimg_opt = to_pil(standard_deviation_weighted_grey_world(from_pil(rawimg), fitted_params))

mse_opt = mse(from_pil(pimg_opt), from_pil(groundtruth))
ssim_opt = ssim(from_pil(pimg_opt), from_pil(groundtruth), multichannel=True)
print('optimized sdwgw - mse : %f, ssim : %f'%(mse_opt, ssim_opt))


# SSIM, the higher, two image more similarity

fig, ((ax1, ax2, _), (ax3, ax4, ax5)) = plt.subplots(2, 3)
ax1.imshow(from_pil(rawimg))
ax1.set_title('input')
ax2.imshow(from_pil(groundtruth))
ax2.set_title('groundtruth')
# ax6.imshow(from_pil(pimg_sdwgw02))
# ax6.set_title('previous results')
ax3.imshow(pimg_gw)
ax3.set_title('grey world, mse: %.02f, ssim:%.02f'%(mse_gw, ssim_gw))
ax4.imshow(pimg_sdwgw)
ax4.set_title('sdwgw init, mse: %.02f, ssim: %.02f'%(mse_sdwgw, ssim_sdwgw))
ax5.imshow(from_pil(pimg_opt))
ax5.set_title('optimization, mse: %.02f, ssim: %.02f'%(mse_opt, ssim_opt))
plt.show()
