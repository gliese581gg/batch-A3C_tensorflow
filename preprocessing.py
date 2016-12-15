import cv2
import numpy as np

def preprocess(params,imag):
	result = imag.copy()
	if params['rom'] != 'toy_way' and (np.array(result.shape)!=np.array([params['img_h'],params['img_w'],params['img_c']])).any():
		result = cv2.resize(result,(84,110))
		result = result[18:102,:,:]

	if params['img_c'] == 1 :  result =  cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
	result = result.reshape((params['img_h'],params['img_w'],params['img_c']))
	return result
		

