#!/bin/python 

import tensorflow as tf
from keras.layers import Input
import zipfile
import os

from steering.bottleneck_generator import BottleneckData
from steering.model import create_model, train_model
from steering.util import upload_s3, zipdir, stop_instance, full_path

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

def put_tensorboard_logs():
  data_folder = full_path('logs')

  print("Zipping folder", data_folder)
  zf = zipfile.ZipFile(zipfile_path, "w")
  zipdir(data_folder, zf)
  zf.close()
  print("Finished zipping folder", data_folder)

  upload_s3(zipfile_name)

class Config(object):
	def __init__(self, batch_size, max_epochs, video_frames, min_delta, patience):
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.min_delta = min_delta
		self.patience = patience
		self.video_frames = video_frames

	def info(self):
		print("batch_size:", self.batch_size)
		print("epochs:", self.max_epochs)
		print("video_frames:", self.video_frames)
		print("min_delta:", self.min_delta)
		print("patience:", self.patience)

	def model_file(self):
		return "model_{}_{}_{}_{}_{}.ckpt".format(self.batch_size, self.max_epochs, self.video_frames, self.min_delta, self.patience)

	def csv_log_file(self):
		return "model_logs_{}_{}_{}_{}_{}.csv".format(self.batch_size, self.max_epochs, self.video_frames, self.min_delta, self.patience)


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_integer('epochs', 1, 'Number of training examples.')
flags.DEFINE_integer('video_frames', 50, 'Number of video frames to include in each cycle.')
flags.DEFINE_float('min_delta', 0.1, 'Early stopping minimum change value.')
flags.DEFINE_integer('patience', 10, 'Early stopping epochs patience to wait before stopping.')
flags.DEFINE_boolean('verbose', False, 'Whether to use verbose logging when constructing the data object.')
flags.DEFINE_boolean('stop', True, 'Stop aws instance after finished running.')
flags.DEFINE_boolean('include_tensorboard', True, 'Save collect and save tensorboard data.')

def main(_):
	config = Config(FLAGS.batch_size, FLAGS.epochs, FLAGS.video_frames, FLAGS.min_delta, FLAGS.patience)
	config.info()

	data = BottleneckData(batch_size=FLAGS.batch_size, video_frames=FLAGS.video_frames, verbose=FLAGS.verbose)

	bottleneck_input = Input(batch_shape=data.bottleneck_shape(), dtype='float32', name='bottleneck_left_right_center')
	vehicle_inputs = Input(batch_shape=data.vehicle_shape(), dtype='float32', name='angle_torque_speed')

	rnn_model = create_model(bottleneck_input, vehicle_inputs, video_frames=FLAGS.video_frames) 
	train_model(rnn_model, data, config, FLAGS.include_tensorboard)
	if FLAGS.include_tensorboard:
		put_tensorboard_logs()

	upload_s3(config.csv_log_file())
	upload_s3(config.model_file())

	if FLAGS.stop:
		stop_instance()

if __name__ == '__main__':
	tf.app.run()
