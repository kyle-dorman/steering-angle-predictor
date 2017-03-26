#!/bin/python

from orig_generator import OrigData
from util import full_path, upload_s3, stop_instance

import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Input
import pickle
import time
import os

def train_bottleneck_features(batch_size, save):
	data = OrigData(batch_size=batch_size)

	model = Sequential()
	model.add(Lambda(lambda x: preprocess_input(x)))
	model.add(VGG16(input_shape=data.shape(), include_top=False))
	model.add(BatchNormalization())

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
