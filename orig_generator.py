#!/bin/python

import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from util import full_path

class OrigData(object):
	def __init__(self, batch_size=32):
		self.batch_size = batch_size
		data_folder = full_path("image_data")
		self.video_folders = [os.path.join(data_folder,child) for child in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder,child))]
		self.generators = self._generators()

	def _generators(self):
		return [VideoGenerator(video_folder, self.batch_size) for video_folder in self.video_folders]

	def shape(self):
		image_shape = self.generators[0].image_shape()
		return [None, image_shape[0], image_shape[1], image_shape[2]]

class VideoGenerator(object):
	def __init__(self, video_folder, batch_size=32):
		self.name = video_folder.split("/")[-1]
		self.video_folder = video_folder
		self.batch_size = batch_size
		#, names=["index","timestamp","width","height","frame_id","filename","angle","torque","speed","lat","long","alt"])
		self.df = pd.read_csv(video_folder + "/interpolated.csv")

		self.batch_index = 0

	def size(self):
		return len(self.df.index) / 3

	def image_shape(self):
		return self.image(0).shape

	def image(self, index):
		path = self.video_folder + "/" + self.df['filename'][index]
		return img_to_array(load_img(path))

	def images(self, indices):
		return np.array([self.image(i) for i in indices])

	def __next__(self):
		return self.next()

	def next(self):
		last_index = len(self.df.index)
		start_index = self.batch_index
		end_index = min(self.batch_index + (3 * self.batch_size), last_index)
		# reset index when we get to the end
		self.batch_index = end_index if end_index < last_index else 0 

		left = []
		right = []
		center = []

		for i in range(start_index, end_index, 3):
			left.append(i)
			right.append(i+1)
			center.append(i+2)

		return [
			self.images(left),
			self.images(right),
			self.images(center)
		]