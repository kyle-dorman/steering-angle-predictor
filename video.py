#!bin/python

import cv2
import argparse
import os
import random
import numpy as np
import time
from collections import deque
from keras import models
from keras.layers import Input
import tensorflow as tf
import keras.backend as K

from result_generator import ImageGenerator
import draw
from model import get_image_processor_model, predict, process_images

def build_video(output, model_file, number_frames, batch_size, dataset, use_gpu):
	video_folder = "image_data/" + dataset
	bottleneck_data_file = "bottleneck_data/" + dataset
	generator = ImageGenerator(dataset, video_folder, number_frames)

	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	_, height, width, _ = generator.frame_images_shape()
	out = cv2.VideoWriter(output, fourcc, 20.0, (width, height), True)

	processed_image_queue = deque()
	vehicle_data_queue = deque()
	processed_image_shape = (3, 7, 10, 512)
	for i in range(number_frames):
		processed_image_queue.append(np.zeros(processed_image_shape))
		vehicle_data_queue.append(np.zeros(generator.vehicle_data_shape()))

	image_buffer = [np.zeros(np.array(processed_image_queue).shape) for i in range(batch_size)]
	vehicle_buffer = [np.zeros(np.array(vehicle_data_queue).shape) for i in range(batch_size)]

	if use_gpu == False:
		config = tf.ConfigProto(device_count = {'GPU': 0})

	with tf.Session(config=config) as sess:
		K.set_session(sess)

		i_width = 320
		i_height = 240
		image_input = Input(shape=(i_height, i_width, 3), dtype='float32', name='image_input')
		image_processor_model = get_image_processor_model(image_input)
		reccurent_model = models.load_model(model_file)

		#skip forward 200 frames
		for i in range(220):
			generator.next()
		# pad queue with good data
		for i in range(number_frames):
			generator.next()
			feature, steering_angle = generator.next()
			start = time.time()
			processed_images = process_images(image_processor_model, feature[0], batch_size, i_width, i_height)
			processed_image_queue.append(processed_images)
			vehicle_data_queue.append(feature[1])
			processed_image_queue.popleft()
			vehicle_data_queue.popleft()

		center_images = []
		predictions = []
		actuals = []
		speeds = []
		image_buffer = []
		vehicle_buffer = []
		start = time.time()

		for i in range(220+number_frames, 220+number_frames + 32):#range(generator.length):
			print("index:", i, "/", generator.length)
			feature, steering_angle = generator.next()
			
			processed_images = process_images(image_processor_model, feature[0], batch_size, i_width, i_height)
			processed_image_queue.append(processed_images)
			vehicle_data_queue.append(feature[1])
			processed_image_queue.popleft()
			vehicle_data_queue.popleft()

			image_buffer.append(np.array(list(processed_image_queue)))
			vehicle_data = np.array(list(vehicle_data_queue))
			for i in range(vehicle_data.shape[0]-6, vehicle_data.shape[0]):
				vehicle_data[i][0] = 0
			vehicle_buffer.append(vehicle_data)
			center_images.append(np.copy(feature[0][2]))
			actuals.append(steering_angle)
			speeds.append(feature[1][2])

		features_list = (np.array(image_buffer), np.array(vehicle_buffer))
		predictions = predict(reccurent_model, features_list)
		print(np.max(predictions))
		print(np.min(predictions))

		end = time.time()
		print("Took {:.2f} seconds to predict steering angle".format(end - start))
		print(predictions)
		# speed = feature[1][2] # what are the units here?

		for i in range(len(predictions)):
			img = center_images[i]
			speed = speeds[i]
			steering_angle = actuals[i]
			predicted_angle = predictions[i]
			error_percentage = (steering_angle - predicted_angle)/steering_angle
			print("Predicted {:.3f} steering angle and actual is {:.3f}. Error percentage is {:.4f}".format(predicted_angle, steering_angle, error_percentage))

		# img = np.copy(feature[0][2])
			img = draw.draw_path_on(img, speed, steering_angle)
			img = draw.draw_path_on(img, speed, predicted_angle, color=(0,255,0))
			img = draw.draw_error(img, error_percentage)
			img = img.astype('u1')
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			out.write(img)

		# Release everything if job is finished
		print("The output video is {}".format(output))
		out.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	# Construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
	parser.add_argument("-m", "--model_file", required=False, default='model_32_100_50_0.001_10.ckpt', help="model file name")
	parser.add_argument("-n", "--number_frames", required=False, default=50, help="Number of video frames to pass to model. Must match size you trained on.")
	parser.add_argument("-b", "--batch_size", required=False, default=32, help="Batch size for the saved trained model. Required for input sizes to match.")
	parser.add_argument("-d", "--dataset", required=False, default="HMB_4", help="Video foulder to build from.")
	parser.add_argument("-g", "--use_gpu", required=False, default=False, help="Whether to run the model with the gpu or not.")
	args = vars(parser.parse_args())

	# Arguments
	output = args['output']
	model_file = args['model_file']
	number_frames = args['number_frames']
	batch_size = args["batch_size"]
	dataset = args['dataset']
	use_gpu = args["use_gpu"]


	build_video(output, model_file, number_frames, batch_size, dataset, use_gpu)


