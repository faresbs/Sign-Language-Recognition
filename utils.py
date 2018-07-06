import numpy as np
import cPickle
import joblib
import cv2, os, os.path
import glob

from keras import utils as keras_utils

def load_data(path):

	files = os.listdir(path)

	outputs = []
	output_names = []

	if (len(files) == 0):
		print "There are no files!"
		return None

	for file in files:
		if os.path.isfile(path+'/'+file) and os.access(path, os.R_OK):
			print "File exists and is readable"
			print "Extracting data.."

			with open(path+'/'+file, "rb") as input_file:
				e = joblib.load(input_file)

			outputs.append(e)
			output_names.append(file)

		else:
			print "Either file is missing or is not readable"

	#To know the order of the outputs
	print output_names

	return outputs
	

#Extract classes names from raw data
def load_names(path):

	files = os.listdir(path)

	if (len(files) == 0):
		print "There are no file!"
		return None

	for file in files:
		if os.path.isfile(path+'/classes_names') and os.access(path, os.R_OK):
			print "File exists and is readable"
			print "Extracting classes names.."

			with open(path+'/classes_names', "rb") as input_file:
				e = joblib.load(input_file)
			print e
			return e



#Load data from many batches
def load_data_from_batches(path, folders):

	output_all_batches = []

	for folder in folders:
		print "\nExtract data from "+ folder+".."
		o = load_data(path+'/'+folder) 
		output_all_batches.append(o)

	return output_all_batches


#Split data to train and test sets
def train_test_split(X, y, test_size):
	ratio = int(X.shape[0]*test_size)
	X_train = X[ratio:,:]
	X_test =  X[:ratio,:]
	y_train = y[ratio:]
	y_test =  y[:ratio]
	return X_train, X_test, y_train, y_test



def BatchGenerator(path):

	y, names, X = load_data(path)
	num_of_classes = len(names)

	#Reshape input data to be fed to the network
	#[number, x, y, channels]
	if (X.ndim == 4):
		X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 3))
	else:
		X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))


	# Making sure your input shape is correct
	assert(X.ndim == 4)
	assert(y.ndim == 1)

	#Convert class vectors to binary class matrices
	y = keras_utils.to_categorical(np.ravel(y), num_of_classes)

	#Normalize data
	print "Normalize data.."
	X = X.astype('float16')
	X /= 255


	#Split data
	print "Splitting data to train and test"
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	
	del X
	del y

	limit = 2000
	start = 0

	while 1:

		end = start + limit

		if(end >= len(X_train)):
			yield X_train[start:len(X_train)], y_train[start:len(X_train)], -1 

		yield X_train[start:end], y_train[start:end], end

		start = end


#PATH ='/home/farris/Projects/Sign-Language-Recognition-System/static image recognition/data'
#y, names, X = load_data(PATH)