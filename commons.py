import io
import torch
from PIL import Image
from .network.tiramisu import *
import numpy as np
import os

#Reading the saved checkpoint
def get_model(is_cuda):
	curr = os.path.dirname(os.path.realpath(__file__))
	checkpoint = os.path.join(curr,'checkpoint.tar')

	net = FCDenseNet103(n_classes=1)
	#Toggle running in gpu or cpu
	if is_cuda:
		state_dict = torch.load(checkpoint,map_location='cuda')
	else:
		state_dict = torch.load(checkpoint,map_location='cpu')
	net.load_state_dict(state_dict['model_state_dict'])
	net.eval()
	if is_cuda:
		net = net.cuda()
	return net

def get_image(image_bytes=None):
	image= Image.open(io.BytesIO(image_bytes))
	return image

#combine and stitch together the results on all the patches 
def attach(p_list, stride,imgwidth,imgheight):
	new_im = Image.new('L', (imgwidth,imgheight))
	new_im = np.asarray(new_im).astype('float')
	for i,r in enumerate(p_list):
		x = stride*i
		for j,patch in enumerate(r):
			y = stride*j 
			patch = np.asarray(patch).astype('float')
			height,width = patch.shape
			try:
				new_im[x:x+height,y:y+width] =  (new_im[x:x+height,y:y+width] + patch)
				# if i!=0:
				# 	new_im[x:x+56,y:y+width]/=2
				# if j!=0:
				# 	new_im[x:x+height,y:y+56]/=2
			except:
				pdb.set_trace()
	new_im = new_im.astype('uint8')
	new_im = Image.fromarray(new_im)
	return new_im

#Crop patches from the given image at a given stride for passsing to the network
def crop(im, height, width,stride=250):
	patchlist = []
	k = 0
	im = np.asarray(im)
	imgheight,imgwidth,_ = im.shape
	for i in range(0,imgheight,stride):
		rlist = []
		for j in range(0,imgwidth,stride):
			x = j+width
			y = i+height
			if x > imgwidth:
				x = imgwidth
			if y > imgheight:
				y = imgheight
			rlist.append(Image.fromarray(im[i:y,j:x,:]))
		patchlist.append(rlist)
	return patchlist




