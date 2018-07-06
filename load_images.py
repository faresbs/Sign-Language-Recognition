"""
Generic script for extracting data of images according to their corresponding folder class
"""

import numpy as np
import cPickle
import joblib
import cv2, os
import glob

np.random.seed(0)


#Return data:[total_samples,image_size, image_size] and labels:[total_samples, 1]
def extract(root, isGray=True):

	temp = []
	classes_names = []

	#Measure the total number images of all the folders 
	size_all = 0

	print 'Reading images..'

	for folder in os.listdir(root):
		images = [cv2.imread(file) for file in glob.glob(root+folder+'/*.png')]
		
		size_all += len(images)

		images = np.array(images)

		if(isGray):
			images = images[:,:,:,0] 
			
		classes_names.append(folder)
		temp.append(images)


	#Extract size image
	size_image = len(temp[0][0])

	#For the image data values
	if (isGray):
		data = np.zeros((size_all, size_image, size_image), dtype=np.uint8)
	else:
		data = np.zeros((size_all, size_image, size_image, 3), dtype=np.uint8)


	#For the label classes
	labels = np.zeros((size_all, ), dtype=np.int32) 

	
	start = 0

	for i in range(len(classes_names)):

		data[start:len(temp[i])+start, :, :] = temp[i]
		labels[start:len(temp[i])+start, ] = i 

		start += len(temp[i])

	return data, labels, classes_names



def shuffle(X, y):
	
	ind_list = [i for i in range(len(y))]
	np.random.shuffle(ind_list)

	if (X.ndim == 4):
		X_new  = X[ind_list, :, :, :]
	else:
		X_new  = X[ind_list, :,:]

	
	y_new = y[ind_list,]

	return X_new, y_new


#Save the numerical data
def save(X, y, names, to_save):

	#If there is no directory,create one
	if not os.path.exists(to_save):
		os.makedirs(to_save)

	print "Saving files.."

	#Save data images
	if os.path.isfile(to_save+'/data_images'):
		print "file exists: "+to_save+'/data_images'
	else:
		with open(to_save+"/data_images", "wb") as file:
			joblib.dump(X, file) 
	
	#Save labels
	if os.path.isfile(to_save+'/labels_images'):
		print "file exists: "+to_save+'/labels_images'
	else:
		with open(to_save+"/labels_images", "wb") as file:
			joblib.dump(y, file)

	#Save classes names
	if os.path.isfile(to_save+'/classes_names'):
		print "file exists: "+to_save+'/classes_names'
	else:
		with open(to_save+"/classes_names", "wb") as file:
			joblib.dump(names, file)


#Save data in many batches
def save_in_batches(root, to_save, isGray=True):

	for folder in os.listdir(root):
		print ('Saving data for batch '+folder)
		X, y, names = extract(root+folder+'/', isGray)
		X, y = shuffle(X, y)
		save(X, y, names, to_save+folder)


#For 1 batch of data

X, y, names = extract('all/', False)
X, y = shuffle(X, y)	
save(X, y, names, 'data/224/ALL')


#save_in_batches('flipped_gray_small_files/ASL/alphabet/', 'data/224/FLIP/', True)