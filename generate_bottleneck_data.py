#!/bin/python

from orig_generator import OrigData
from util import full_path, upload_s3, stop_instance

import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
import pickle
import time
import os

def train_bottleneck_features(batch_size, save):
	data = OrigData(batch_size=batch_size)
	zipfile_name = 'tensorboard.zip'
	zipfile_path = full_path(zipfile_name)
	inputs = Input(shape=data.shape())
	# create the base pre-trained model
	base_model = VGG16(input_tensor=inputs, include_top=False)
	output = BatchNormalization()(base_model.output)

	model = Model(input=inputs, output=output)

	print('Bottleneck training')

	for generator in data.generators:
		results = {'left': [], 'right': [], 'center':[]}

		for direction in ['left', 'right', 'center']:
			t = time.time()
			print("Generating bottleneck data for generator:", generator.name, "and direction:", direction)
			generator.set_direction(direction)
			results[direction] = model.predict_generator(generator, generator.size())
			print("Done generatring output. Took", time.time() - t, "seconds.")

		file = "bottleneck_data/" + generator.name + ".p"
		pickle.dump(results, open(full_path(file), 'wb'))

		if save:
			print("Saving files", file)
			upload_s3(file)
			os.remove(full_path(file))
			print("Removed", file)
		else:
			print("Not saving files.")

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_boolean('save', True, 'Save the generated bottleneck model to S3.')
flags.DEFINE_boolean('stop', True, 'Stop aws instance after finished running.')

def main(_):
	print("Using batchsize", FLAGS.batch_size)
	train_bottleneck_features(FLAGS.batch_size, FLAGS.save)

	if FLAGS.stop:
		stop_instance()

if __name__ == '__main__':
	tf.app.run()
