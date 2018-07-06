import numpy as np
import cv2, os
import errno 
import datetime as dt
from time import time
import joblib

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import utils as keras_utils
import keras.applications.vgg16 as vgg16
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception

from matplotlib import pyplot as plt

import utils
import cnn_models


np.random.seed(0)


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))



def train(path, tune=True):

	#When the model started training
	start_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")
	print ("Start Time: "+start_date)

	#Load model architecture
	if(tune==True):
		print "pretrained model loaded.."
		vgg_model = vgg16.VGG16(weights='imagenet')
	else:
		vgg_model = vgg16.VGG16()
		#model = cnn_models.custom_model(num_of_classes, image_size, weights_path=False)
	
	model = Sequential()
	for i in range(len(vgg_model.layers)-3):
		model.add(vgg_model.layers[i])


	model.add(Dense(4096, kernel_initializer='uniform', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, kernel_initializer='uniform', activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(24, activation='softmax'))

	if (tune==True):
		# set the first n layers to non-trainable (weights will not be updated)
		for layer in model.layers[:15]:
			layer.trainable = False


	#Print out the model structure
	print(model.summary())


	#Train model using batchs using flow keras

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

	batch_size = 32


	train_generator = train_datagen.flow_from_directory(
        path+'/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

	# same es the train_generator    
	validation_generator = test_datagen.flow_from_directory(
	        path+'/validation',
	        target_size=(224, 224),
	        batch_size=batch_size,
	        class_mode='categorical',
	        shuffle=True)


	sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
			        
	#Debug with tensorboard
	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	print "Training.."

	#when augmenting the data, you need to stretch by multiplying
	history = model.fit_generator(
	        train_generator,
	        steps_per_epoch=len(train_generator) * 2,
	        epochs=10,
	        verbose=1,
	        validation_data=validation_generator,
	        validation_steps=len(validation_generator),
	        callbacks=[tensorboard])



	#Evaluation of the model
	print "Evaluating the model with val data.."
	scores = model.evaluate_generator(generator=validation_generator, verbose=1)
	print("Accuracy: %.2f%%" % (scores[1]*100))


	#date of the model stoped Training
	end_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")

	print ("Start Time: "+start_date)
	print("End Time: "+end_date)


	print "Display learning curve.."
	# Plot the accuracy and loss curves
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'g', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig("accuracy_"+end_date+".png")

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'g', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig("loss_"+end_date+".png")


	#plt.show()


	#Save history
	with open("history_"+end_date, "wb") as file:
		joblib.dump(history.history, file) 


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

	limit_mem()

	return None


#Predict with new examples
def predict(a):

	classes = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

	path_weights = 'weights/2018-07-06-12:59'

	loaded_model = cnn_models.load_pre_tune_model(24, path_weights)

	a = np.reshape(a, (1, a.shape[0], a.shape[1], 3))

	a = a.astype('float16')
	a /= 255

	y_prob = loaded_model.predict(a) 
	prediction = classes[y_prob.argmax(axis=-1)[0]]

	print prediction

	return prediction


"""
PATH = 'small_files/ASL'
train(PATH)
"""

classes = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

#image = cv2.imread('testa1.jpg')
image = cv2.imread('flipped_small_files/ASL/alphabet/A/a/1000.png')
#image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

image = cv2.resize(image, (224,224))

PATH = 'weights/2018-07-06-12:59'

predict(image)
