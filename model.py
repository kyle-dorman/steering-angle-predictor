#!/bin/python

from orig_generator import OrigData
from util import full_path, upload_s3

from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.layers import Input
import pickle

def train_bottleneck_features(batch_size, save):
	data = OrigData(batch_size=batch_size)
	generators = data.generators()

	inputs = Input(shape=generators[0].image_shape())
	vgg = VGG16(input_tensor=inputs, include_top=False)
	model = BatchNormalization()(vgg)

	print('Bottleneck training')

	files = []

	for generator in generators:
		output_file = full_path("bottleneck_data/" + generator.name + ".p")
		files.append(output_file)
		
		bottleneck_features = model.predict_generator(generator, generator.size())
		pickle.dump(bottleneck_features, open(output_file, 'wb'))

	if save:
		print("Saving files.")
		save_bottleneck_features(files)
	else:
		print("Not saving files.")

def save_bottleneck_features(files):
	for file in files:
		file = file.split("steering-angle-predictor/")[-1]
		upload_s3(file)
