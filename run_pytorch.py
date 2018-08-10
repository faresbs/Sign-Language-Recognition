import torch 
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
from model_pytorch import network
import joblib

np.random.seed(0)


def get_test_input(image, input_dim, CUDA):
    img = cv2.imread(image)
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    
    return img_



if __name__ == '__main__':

	#why output prediction changes when using the same image

	
	path_weights = 'weights/2018-07-18-11:01/weights.h5'

	#Saving class names
	with open('weights/2018-07-18-11:01/class_names', "rb") as file:
		class_names = joblib.load(file)

	num_classes = len(class_names)

	print('class_names: '+str(class_names))

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print (device)

	print("Loading network.....")
	model = network(num_classes)
	model.load_state_dict(torch.load(path_weights))
	print("Network successfully loaded")

	#This takes a long time (it is done one time)
	model = model.to(device)

	img = get_test_input('testa.jpg', 224, True)
	img = img.to(device)

	import timeit

	start = timeit.default_timer()
	output = model(img)
	prediction = output.data.argmax()
	_, predicted = torch.max(output.data, 1)
	print (class_names[predicted])

	stop = timeit.default_timer()
	print (stop - start) 