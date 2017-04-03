#!/bin/python 

import numpy as np
import os
import glob 
import random
import pandas as pd
from collections import deque

from util import full_path, open_pickle_file, open_large_pickle_file

class BottleneckData(object):
	def __init__(self, batch_size=32, video_frames=100, verbose=False):
		self.batch_size = batch_size
		self.video_frames = video_frames
		image_data = full_path("image_data")

		self.video_datasets = {}
		for child in os.listdir(image_data): 
			if os.path.isdir(os.path.join(image_data,child)) == False: continue

			self.video_datasets[child] = (os.path.join(image_data,child) , "bottleneck_data/{}.p".format(child))

		self.datasets = [VideoDataset("HMB_1", "image_data/HMB_1", "bottleneck_data/HMB_1.p", self.batch_size, self.video_frames, verbose=verbose)]
		# self.datasets = [VideoDataset(key, video_data[0], video_data[1], self.batch_size, self.video_frames, verbose=verbose) for key, video_data in self.video_datasets.items()]

	def bottleneck_shape(self):
		return self.datasets[0].bottleneck_shape()

	def vehicle_shape(self):
		return self.datasets[0].vehicle_shape()

class VideoDataset(object):
	def __init__(self, name, video_folder, bottleneck_data_file, batch_size, video_frames, verbose=False):
		self.name = name
		self.video_folder = video_folder
		self.bottleneck_data_file = bottleneck_data_file
		self.batch_size = batch_size
		self.video_frames = video_frames
		self.df = pd.read_csv(video_folder + "/interpolated.csv")
		if os.path.getsize(bottleneck_data_file) > 4000000000:
			self._bottleneck_data = open_large_pickle_file(bottleneck_data_file)
		else:
			self._bottleneck_data = open_pickle_file(bottleneck_data_file)

		self.video_length = len(self.df.index) // 3
		self.valid_generators = []
		self.train_generators = []
		self.reset_generators()

		if verbose:
			self.info()

	def reset_generators(self):
		MIN_IMAGE_SEQUENCE = 600 # 30 seconds as 20 fps
		valid_sequence_length = max(self.video_length // 5, MIN_IMAGE_SEQUENCE)
		split_index = random.randint(MIN_IMAGE_SEQUENCE + self.video_frames, self.video_length - MIN_IMAGE_SEQUENCE - valid_sequence_length - 1 - self.video_frames)

		train_1 = BottleneckDataIterator(self.batch_size, self.video_frames, 0, split_index, self)
		valid_start = split_index + 1
		valid_end = valid_start + valid_sequence_length
		valid_1 = BottleneckDataIterator(self.batch_size, self.video_frames, valid_start, valid_end, self)
		train_2_start = valid_end + 1
		train_2 = BottleneckDataIterator(self.batch_size, self.video_frames, valid_end + 1, self.video_length - 1, self)

		self.valid_generators = [valid_1]
		self.train_generators = [train_1, train_2]

	def bottleneck_data(self, index):
		return np.array([self._bottleneck_data['left'][index], self._bottleneck_data['right'][index], self._bottleneck_data['center'][index]])

	def bottleneck_shape(self):
		return [self.batch_size, self.video_frames] + list(self.bottleneck_data(self.video_length-1).shape)

	def vehicle_data(self, index):
		return self.df.iloc[index][6:9].values

	def vehicle_shape(self):
		return [self.batch_size, self.video_frames] + list(self.vehicle_data(self.video_length-1).shape)

	def steering_angle(self, index):
		return self.df.iloc[index][6]

	def info(self):
		print("Dataset", self.name, "is size", self.video_length)
		print("Bottleneck shape:", self.bottleneck_shape(), "vehicle shape:", self.vehicle_shape())
		print("Valid: start:", self.valid_generators[0].start_index, "end:", self.valid_generators[0].end_index)
		print("Test 1 start:", self.train_generators[0].start_index, "end:", self.train_generators[0].end_index)
		print("Test 2 start:", self.train_generators[1].start_index, "end:", self.train_generators[1].end_index)

class BottleneckDataIterator(object):
	def __init__(self, batch_size, video_frames, start_index, end_index, dataset):
		self.batch_size = batch_size
		self.video_frames = video_frames
		self.start_index = start_index
		self.end_index = start_index + (((end_index - start_index)//batch_size)* batch_size)
		self.batch_index = start_index
		self.dataset = dataset
		self.bottleneck_queue = deque()
		self.vehicle_data_queue = deque()
		for i in range(video_frames):
			self.bottleneck_queue.append(np.zeros(self.dataset.bottleneck_shape()[2:]))
			self.vehicle_data_queue.append(np.zeros(self.dataset.vehicle_shape()[2:]))

	def __next__(self):
		return self.next()

	def next(self):
		last_index = self.end_index
		start_index = self.batch_index
		end_index = min(start_index + self.batch_size, last_index)
		# reset index when we get to the end
		self.batch_index = end_index if end_index < last_index else start_index

		result = {'bottleneck_left_right_center': [], 'angle_torque_speed': [] }
		labels = []
		for index in range(start_index, end_index):
			self.bottleneck_queue.append(self.dataset.bottleneck_data(index))
			self.vehicle_data_queue.append(self.dataset.vehicle_data((3*index)+2)) # csv has left, right, center
			self.bottleneck_queue.popleft()
			self.vehicle_data_queue.popleft()

			result['bottleneck_left_right_center'].append(list(self.bottleneck_queue))
			result['angle_torque_speed'].append(list(self.vehicle_data_queue))
			labels.append([self.dataset.steering_angle((3*index)+2)])

		result['bottleneck_left_right_center'] = np.array(result['bottleneck_left_right_center'])
		result['angle_torque_speed'] = np.array(result['angle_torque_speed'])

		length = len(result['angle_torque_speed'])
		angles_to_remove = min(6, self.video_frames)
		for i in range(length-angles_to_remove, length):
			### don't expose speed for the last 5 frames
			result['angle_torque_speed'][i][0] = 0
		return (result, np.array(labels))

	def size(self):
		return self.end_index - self.start_index
