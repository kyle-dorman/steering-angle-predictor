#!/bin/python 

import tensorflow as tf
from keras.layers import Input

from bottleneck_generator import BottleneckData
from model import create_model, train_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_integer('epochs', 1, 'Number of training examples.')
flags.DEFINE_integer('video_frames', 50, 'Number of video frames to include in each cycle.')

def main(_):
	print("Using batch_size", FLAGS.batch_size)
	print("Running", FLAGS.epochs, "epochs")
	print("Using", FLAGS.video_frames, "video_frames")

	data = BottleneckData(batch_size=FLAGS.batch_size, video_frames=FLAGS.video_frames)

	bottleneck_input = Input(batch_shape=data.bottleneck_shape(), dtype='float32', name='bottleneck_left_right_center')
	vehicle_inputs = Input(batch_shape=data.vehicle_shape(), dtype='float32', name='angle_torque_speed')

	rnn_model = create_model(bottleneck_input, vehicle_inputs, video_frames=FLAGS.video_frames) 
	train_model(rnn_model, data, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, video_frames=FLAGS.video_frames)

if __name__ == '__main__':
	tf.app.run()