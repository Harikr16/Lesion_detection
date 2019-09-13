from .commons import *
import torch
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import pdb
from PIL import Image

is_cuda = True
net = get_model(is_cuda)
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd =  [0.24703233, 0.24348505, 0.26158768]


def superimpose(image,mask):
#superimpose cancer on wsi image
	alpha = 0.8
	superimposed = np.asarray(image).copy()
	superimposed[mask==255] = (0,0,255)
	superimposed = superimposed*alpha + np.asarray(image).copy()*(1-alpha)
	return superimposed
	

#To give the lesion prediction of network on the given image 
def get_cancer_prediction(image_bytes):
	with torch.no_grad():
		img = get_image(image_bytes)
		imwidth,imheight = img.size 
		plist = crop(img, 256, 256,200)
		outlist = []
		for rlist in plist:
			out = []
			for i in range(len(rlist)):
				rlist[i] = ToTensor()(rlist[i])
				rlist[i] = TF.normalize(rlist[i],normMean,normStd)
				if is_cuda:
					rlist[i] = rlist[i].unsqueeze(0).cuda()
				image = rlist[i]
				output = net(image)
				output = ToPILImage()(output.cpu().squeeze(0))
				out.append(output)
			outlist.append(out)
		outimg = attach(outlist,200,imwidth,imheight)
		outimg = np.array(outimg)
		outimg = cv2.resize(outimg,(imwidth,imheight))

		low_conf_pixels = outimg[outimg>65]
		low_conf_pixels = low_conf_pixels[low_conf_pixels<190]
		low_conf_pixels_count = len(low_conf_pixels)
		confidence = 1 - (low_conf_pixels_count/imwidth*imheight)
		_,outimg = cv2.threshold(outimg,127,255,cv2.THRESH_BINARY)
	
	flag=1
	# Criterion for presence of lesion
	if len(outimg==255)<100:
		flag == 0
	superimposed = superimpose(img,outimg)
	superimposed = Image.fromarray(superimposed.astype('uint8'))
	return superimposed,confidence,flag

if __name__ == '__main__':
	validate()