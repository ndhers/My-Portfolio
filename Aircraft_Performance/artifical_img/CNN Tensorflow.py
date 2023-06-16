#!/usr/bin/env python
#-*- coding: utf-8 -*- 

import cv2
import tensorflow as tf
import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
#import random
#import gc
import math
import sklearn
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

X = []
Y = []

nrows = 150
ncolumns = 150

data_dir = '/home/nicodhe/scratch/Database'

CSV_dir = '/home/nicodhe/projects/def-nadaraja/nicodhe/CSV'

data_img = ['/home/nicodhe/scratch/Database/{}'.format(i) for i in os.listdir(data_dir)]

data_csv = ['/home/nicodhe/projects/def-nadaraja/nicodhe/CSV/{}'.format(i) for i in os.listdir(CSV_dir)]

keys = ['0.0','0.025','0.05','0.075','0.1','0.125','0.150','0.175','0.2']
dict_of_mach = dict.fromkeys(keys)

dict_of_lists = dict.fromkeys(keys)	

data_per_airfoil = dict.fromkeys(keys)

data_floating_airfoil = dict.fromkeys(keys)

def getList(dict): 
	return dict.keys()

i = 0

for mach in keys:

	dict_of_lists[mach] = {}
	

	path = 'not working'
	for i in data_csv:
		if '/'+mach+'.csv' in i:
			path = i

	dict_of_mach[mach] = pd.read_csv(path, encoding = "ISO-8859-1")
	

	for column_name in dict_of_mach[mach].columns:
		temp_list = dict_of_mach[mach][column_name].tolist()
		dict_of_lists[mach][column_name] = temp_list

	
		
	aircraft_list = getList(dict_of_lists[mach])
	

	data_per_airfoil[mach] = {}
	#data_per_airfoil[mach].fromkeys(dict_of_lists[mach].keys(),[])


	for name in aircraft_list:
	  
	  data_list = []

	  while('  ') in dict_of_lists[mach][name]: 
	    dict_of_lists[mach][name].remove('  ') 

	    dict_of_lists2 = dict_of_lists

	  for index in range(0,len(dict_of_lists[mach][name])):
	  	try:
	  		p = float(dict_of_lists2[mach][name][index])
	  		continue
	  	except:
	  		data_list_append = dict_of_lists[mach][name][index]
	  		data_list.append(data_list_append)
	  data_per_airfoil[mach][name] = data_list
	
	
	data_floating_airfoil[mach] = {}


	for name in aircraft_list:
	  listtest = []
	  listtest2 = []
	  data_floating_airfoil[mach][name] = []
	  for index in range(0,len(data_per_airfoil[mach][name])):
	  	try:
	  		k = data_per_airfoil[mach][name][index].split()
	  		x = k[0]
	  		y = float(k[1])
	  		z = float(k[2])
	  		listtest = [x,y,z]
	  		listtest2.append(listtest)	  		
	  	except:
	  		pass
	  data_floating_airfoil[mach][name] = listtest2

	  #keep_for_final[name] = listtest2


	for name in aircraft_list:

		if name == 'ï»¿m1':
			name2 = 'm1'
		else:
			name2 = name

		if mach == '0.0':
			mach2 = '0.0.'
		elif mach == '0.1':
			mach2 = '0.1.'
		else:
			mach2 = mach

		for index in range (0,len(data_floating_airfoil[mach][name])):

			lookup = data_floating_airfoil[mach][name][index][0]

			for image in data_img:

				if name2+'_'+lookup+'_'+mach2 in image:

					X.append(cv2.resize(cv2.imread(image,0),(nrows,ncolumns)))
					list1 = [data_floating_airfoil[mach][name][index][1],data_floating_airfoil[mach][name][index][2]]
					Y.append(list1)
					

	#print(mach+' is done')




X = np.array(X)
Y = np.array(Y)


#print('Shape of image dataset is ', X.shape)
#print('Shape of labels is ', Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.20, random_state = 2) 


#print('Shape of train data is ',X_train.shape)
#print('Shape of train labels is ',y_train.shape)
#print('Shape of test data is ',X_test.shape)
#print('Shape of test labels is ',y_test.shape)

X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)

ntrain = len(X_train)
nval = len(X_test)

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (150,150,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(2))


#model.summary()

opt = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=opt,loss='mse',metrics=['mae', 'mse'])

# data augmentation:

train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 32

train_generator = train_datagen.flow(X_train,y_train, batch_size)
val_generator = val_datagen.flow(X_test,y_test, batch_size)

history = model.fit(train_generator, steps_per_epoch = ntrain // batch_size,
	epochs = 50, validation_data = val_generator, validation_steps = nval // batch_size)

#print(val_generator[0])





