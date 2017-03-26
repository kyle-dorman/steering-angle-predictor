#!/bin/python

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import TimeDistributed

from util impor download_s3

def model(load_saved=False):
	if load_saved:
		#
	else:
		create_model()


def create_model():
	# This returns a tensor
	inputs = Input(shape=(784,))

	# a layer instance is callable on a tensor, and returns a tensor
	x = Dense(64, activation='relu')(inputs)
	x = Dense(64, activation='relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=inputs, outputs=predictions)

	# left, right, center outputs of VGG16
	main_input = Input(shape=(3, ?, ?, ?), dtype='int32', name='bottleneck_left_right_center')


def download_bottleneck_features():
	for i in [1,2,4,5,6]:
		download_s3("bottleneck_data/HMB_{}.p".format(i))
