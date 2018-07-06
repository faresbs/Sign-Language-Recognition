from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import cv2, numpy as np
from keras.models import load_model

import keras.applications.vgg16 as vgg16


def custom_model(num_of_classes, image_size, weights_path=False):
   
    #SIMILAR TO VGG16 BUT POOL AND STRIDES ARE (3,3)

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(image_size,image_size,3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))

    
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_of_classes, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model



#VGG_16 in Keras 
#input image is 224x224

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

#AlexNet with batch normalization in Keras 
#input image is 224x224
"""
def AlexNet(weights_path=None):
    model = Sequential()
    model.add(Conv2D(64, 3, 11, 11, border_mode='full'))
    model.add(BatchNormalization((64,226,226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Conv2D(128, 64, 7, 7, border_mode='full'))
    model.add(BatchNormalization((128,115,115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Conv2D(192, 128, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,112,112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Conv2D(256, 192, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,108,108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(12*12*256, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('softmax'))


    if weights_path:
        model.load_weights(weights_path)

    return model
"""

def load_pre_tune_model(num_of_classes, weights_path=None):
    vgg_model = vgg16.VGG16()

    #Remove last layer of the vgg model
    #model.pop()
    #model.outputs = [model.layers[-1].output]
    #model.layers[-1].outbound_nodes = []

    model = Sequential()
    for i in range(len(vgg_model.layers)-3):
        model.add(vgg_model.layers[i])


    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_of_classes, activation='softmax'))
    #model = load_model(weights_path+'/cnn_model.yaml')

    if weights_path:
        model.load_weights(weights_path+'/cnn_weights.h5', by_name=True)

    return model
