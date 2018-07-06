import numpy as np
import cv2, os, shutil
import glob
from distutils.dir_util import copy_tree

alphabet = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

# Resize all images with the same shape
def resize(filename, size):

	images = [cv2.imread(file) for file in glob.glob(filename+'*.png')]

	if not os.path.exists('small_'+filename):
	   os.makedirs('small_'+filename)

	pic_num = 1

	for img in images:
		resized_image = cv2.resize(img,(size, size)) 
		#Save the result on disk in the "small" folder
		cv2.imwrite('small_'+filename+str(pic_num)+'.png',resized_image)
		pic_num += 1

	resized_images = [cv2.imread(file) for file in glob.glob('small_'+filename+'*.png')]
	resized_images = np.array(resized_images)


def resize_all(filename, size):

	print "Resizing.."
	#For all the alphabet
	for letter in alphabet:
		resize(filename+letter+'/', size)


def grayscale(filename):
	grayscale_images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_RGB2GRAY) for file in glob.glob(filename+'*.png')]

	if not os.path.exists('gray_'+filename):
	   os.makedirs('gray_'+filename)

	pic_num = 1

	for img in grayscale_images:

		#Save the grayscale image on disk in the "gray" folder
		cv2.imwrite('gray_'+filename+str(pic_num)+'.png',img)
		pic_num += 1



#For all the alphabet
def grayscale_all(filename):

	print "Transforming to grayscale.."
	for letter in alphabet:
		grayscale(filename+letter+'/')


#Flip images horizontally, vertically or both
def flip_images(filename, axis='vertical'):

	# flip img horizontally
	if(axis == 'horizontal'):
		flipped_images = [cv2.flip(cv2.imread(file), 0) for file in glob.glob(filename+'*.png')]
	# flip img vertically
	if(axis == 'vertical'):
		flipped_images = [cv2.flip(cv2.imread(file), 1) for file in glob.glob(filename+'*.png')]

	# flip img both axis
	if(axis == 'both'):
		flipped_images = [cv2.flip(cv2.imread(file), -1) for file in glob.glob(filename+'*.png')]

	#Count the number of files existing in filename
	pic_num = len([f for f in os.listdir(filename) if os.path.isfile(os.path.join(filename, f))]) + 1

	for img in flipped_images:

		#Save the flipped image on disk in the "flipped" folder
		cv2.imwrite('flipped_'+filename+str(pic_num)+'.png',img)
		pic_num += 1


#For all the alphabet
def flipped_all(filename, axis, augmentation=True):

	#Save original images with the flipped ones
	if(augmentation):
		fromDirectory = filename
		toDirectory = 'flipped_'+filename
		copy_tree(fromDirectory, toDirectory)

	print "flipping images.."
	for letter in alphabet:
		flip_images(filename+letter+'/')



#Regroup all batch images in one folder
def regroup(path):

	folders = os.listdir(path)

	if (len(folders) == 0):
		print "There are no folder!"
		return None

	#Assuming that all folders have the same structure
	#subfolder represent the class name
	#every subfolder have class images
	subfolders = os.listdir(path+'/'+folders[0])

	print "Saving images of batch "+ folders[0]

	# copy subdirectory tree with contents
	fromDirectory = path+'/'+folders[0]
	toDirectory = 'all'

	copy_tree(fromDirectory, toDirectory)

	numfiles = []
	for subfolder in subfolders:
		numfiles.append(len([f for f in os.listdir(path+'/'+folders[0]+'/'+subfolder) if os.path.isfile(os.path.join(path+'/'+folders[0]+'/'+subfolder, f))]))

	n = 0
	for i in range(1, len(folders)):
		print "Saving images of batch "+ folders[i]
		for j in range(len(subfolders)):
			images = [cv2.imread(file) for file in glob.glob(path+'/'+folders[i]+'/'+subfolders[j]+'/*.png')]

			pic_num = numfiles[j] + 1

			for img in images:
				cv2.imwrite('all/'+subfolders[j]+'/'+str(pic_num)+'.png',img)
				pic_num += 1
			numfiles[j] += pic_num


if __name__ == "__main__":
	
	#path = "files/ASL/alphabet"
	
	#folders = os.listdir(path)

	#lEAVE ONE FOLDER FOR TESTING 'E'
	folders = ['A','B','C','D'] 
	#folders = ['B'] 
	#Size of image to resize
	size_image = 224

	#Resize all images with the same size
	#for i in folders: 
	#	print "Preprocessing folder " + i+".."
	#	resize_all('files/ASL/alphabet/' + i +'/' ,size_image)
	#	grayscale_all('small_files/ASL/alphabet/' + i +'/')
	#	flipped_all('small_files/ASL/alphabet/' + i + '/', 'vertical')
	

	regroup("small_files/ASL/alphabet")