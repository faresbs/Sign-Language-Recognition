import numpy as np
import cv2, os
import errno 
import datetime as dt
from time import time

#python garbage collector
import gc

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_yaml
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import utils as keras_utils


import keras.applications.vgg16 as vgg16
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception
import utils
import cnn_models

from matplotlib import pyplot as plt
#For parallel computing
import dask.array as da

np.random.seed(0)

def train(path):
	#Are there many batches or just 1
	#If there are subfolders in path then assume that those are data batches
	#The reason behind using these data batches is to avoid having memory error while loading data to the model

	batches = os.listdir(path)

	previous = None
	loaded_model = None

	#If contains file instead of dir
	if(os.path.isfile(path+'/'+batches[0])):
		cnn_model(path)

	else:
		print "There are "+str(len(batches))+" batches"
		# loop through all the files and folders
		for batch in batches:
			print "Training and testing using batch "+batch+"..\n"
			# check whether the current object is a folder or not
			#if os.path.isdir(os.path.join(path+'/'+batch,file)):
			if previous != None:
				print "Loading previous weights and model.."
				loaded_model = cnn_models.load_pre_tune_model(24, previous)

			previous = cnn_model(path+'/'+batch, loaded_model)



def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

    for i in range(0, 9):
    	gc.collect()


def cnn_model(path, model=None):

	#When the model started training
	start_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")
	print ("Start Time: "+start_date)

	#y, names, X = utils.load_data(path)
	#num_of_classes = len(names)

	#Reshape input data to be fed to the network
	#[number, x, y, channels]
	#if (X.ndim == 4):
	#	X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 3))
	#else:
	#	X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))


	# Making sure your input shape is correct
	#assert(X.ndim == 4)
	#assert(y.ndim == 1)

	#Get image size
	#image_size = X.shape[1]

	#Split Data to chunks using dask
	#print "Splitting to chunks.."
	#k2 = da.from_array(X, chunks = 2)
	#k2.to_delayed()


	#PROBLEM HERE: THIS MODEL IS DIFFERENT THAN THE ONE IN KERAS
	#vgg_model = cnn_models.VGG_16("weights/vgg16_weights.h5")

	if (model == None):
		vgg_model = vgg16.VGG16(weights='imagenet')
	#	model = cnn_models.custom_model(num_of_classes, image_size, weights_path=False)
		print "pretrained model loaded.."

		model = Sequential()
		for i in range(len(vgg_model.layers)-3):
			model.add(vgg_model.layers[i])


		model.add(Dense(4096, kernel_initializer='uniform', activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4096, kernel_initializer='uniform', activation='relu'))
		model.add(Dropout(0.5))

		model.add(Dense(24, activation='softmax'))

		# set the first n layers to non-trainable (weights will not be updated)
		for layer in model.layers[:14]:
			layer.trainable = False

	#Remove last layer of the vgg model
	#model.pop()
	#model.outputs = [model.layers[-1].output]
	#model.layers[-1].outbound_nodes = []

	#model = cnn_models.custom_model(num_of_classes, image_size, weights)

	# check the layers by name : same as using model.summary
	#for i,layer in enumerate(model.layers):
	#    print(i,layer.name)
	#    print(layer.get_output_at(0).get_shape().as_list())

	#Print out the model structure
	print(model.summary())

	#Convert class vectors to binary class matrices
	#y = keras_utils.to_categorical(np.ravel(y), num_of_classes)

	#Normalize data
	#print "Normalize data.."
	#X = X.astype('float16')
	#X /= 255


	#Split data
	#print "Splitting data to train and test"
	#X_train, X_val, y_train, y_val = utils.train_test_split(X, y, test_size=0.65)
	
	#del X
	#del y

	#TRAIN MODEL WITH CHUNKS OF DATA

	#Data augmentation (shift + flip + rotation)	
	train_datagen = ImageDataGenerator(width_shift_range=.2, 
		                             height_shift_range=.2,
		                             horizontal_flip=True,
		                             rotation_range=25,
		                             rescale=1./255
		                             )

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)


	train_generator = train_datagen.flow_from_directory(
        path+'/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

	# same es the train_generator    
	validation_generator = test_datagen.flow_from_directory(
	        path+'/validation',
	        target_size=(224, 224),
	        batch_size=32,
	        class_mode='categorical',
	        shuffle=True)


	



	#fit parameters from data
	#datagen.fit(X_train)

	sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
			        
	#Debug with tensorboard
	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


	#fits the model on batches with real-time data augmentation:
	#model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
	#	                    steps_per_epoch=len(X_train) / 32, epochs=epochs, callbacks=[tensorboard])

	# loads sequentially images and feeds them to the model. 
	# the batch size is set in the constructor 
	model.fit_generator(
	        train_generator,
	        samples_per_epoch=2000,
	        nb_epoch=1,
	        validation_data=validation_generator,
	        nb_val_samples=800)
		
	
	#Evaluation of the model
	print "Evaluating the model with val data.."
	scores = model.evaluate(X_val, y_val, verbose=1,  batch_size=64)
	print("Accuracy: %.2f%%" % (scores[1]*100))


	#Optimizer
	#sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

	#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	
	#Debug with tensorboard
	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	#model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, shuffle=True, validation_data=(X_val, y_val), callbacks=[tensorboard])



	#date of the model stoped Training
	end_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")

	print ("Start Time: "+start_date)
	print("End Time: "+end_date)

	#Make a directory to save the weights
	try:
		os.makedirs('weights')
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

	#Make a directory to save the weights according to date
	try:
		os.makedirs('weights/'+end_date)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
	
		
	#Serialize weights to HDF5
	model.save_weights('weights/'+end_date+"/cnn_weights.h5")


	#Serialize model to YAML
	model_yaml = model.to_yaml()

	#Save model
	with open("weights/"+end_date+"/cnn_model.yaml", "w") as yaml_file:
		yaml_file.write(model_yaml)

	print("Saved CNN model with weights to disk")

	#Delete model to free GPU memory
	print "Deleting current model"

	del model
	limit_mem()

	return ('weights/'+end_date)

#Predict with new examples
def predict(a, path_weights):

	PATH = 'small_files/ASL'
	names = utils.load_names(PATH)

	# load yaml and create model
	#yaml_file = open(path_weights+'/cnn_model.yaml', 'r')
	#loaded_model_yaml = yaml_file.read()
	#yaml_file.close()
	#loaded_model = model_from_yaml(loaded_model_yaml)

	# load weights into new model
	#loaded_model.load_weights(path_weights+"/cnn_weights.h5")
	#print("Loaded model from disk")
	#a = a[:,:,0]

	loaded_model = cnn_models.load_pre_tune_model(len(names), path_weights)

	
	a = np.reshape(a, (1, a.shape[0], a.shape[1], 3))



	y, names, X = utils.load_data("data/224/RGB/test")
	
	#Convert class vectors to binary class matrices
	y = keras_utils.to_categorical(np.ravel(y), len(names))

	sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	
	X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 3))

	#Normalize data
	print "Normalize data.."
	X = X.astype('float16')
	X /= 255

	#Evaluation of the model
	print "Evaluating the model.."
	scores = loaded_model.evaluate(X, y, verbose=1,  batch_size=64)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	
	#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	#loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	#print loaded_model.predict(a)
	print loaded_model.predict_classes(a)
	prediction = names[int(loaded_model.predict_classes(a))]
	print prediction

	return prediction


#PROBLEM WHILE EVALUATING THE MODEL
def evaluate():
	pass

	

PATH = 'small_files/ASL'
cnn_model(PATH)
"""

#image = cv2.imread('testa.jpg')
image = cv2.imread('small_files/ASL/alphabet/C/g/199.png')
#image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

image = cv2.resize(image, (224,224))

PATH = 'weights/2018-07-05-00:35'

predict(image, PATH)
"""