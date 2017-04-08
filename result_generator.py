#!/bin/python 

import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from util import full_path

class ImageGenerator(object):
	def __init__(self, name, video_folder, video_frames):
		self.df = pd.read_csv(video_folder + "/interpolated.csv")
		self.length = (len(self.df.index) // 3) - video_frames
		self.video_folder = video_folder

		# determine the left, right, center order from the first 3 elements
		frame_order = np.array([frame.split("_")[0] for frame in self.df['frame_id'][0:3].values])
		self.left = np.where(frame_order=="left")[0][0]
		self.right = np.where(frame_order=="right")[0][0]
		self.center = np.where(frame_order=="center")[0][0]

		self.index = 0

	def next(self):
		# features = (np.array(list(self.image_queue)), np.array(list(self.vehicle_data_queue)))
		features = (self.frame_images(self.index), self.vehicle_data(self.index))

		result = (features, self.label(self.index))
		self.index += 3
		return result

	def label(self, index):
		return self.df.iloc[index][6]

	def vehicle_data(self, index):
		# skip steering angle
		return np.array([0.0] + list(self.df.iloc[index][7:9].values))

	def vehicle_data_shape(self):
		return self.vehicle_data(0).shape

	def frame_images(self, index):
		return np.array([self.image(index + direction) for direction in [self.left, self.right, self.center]])

	def frame_images_shape(self):
		return self.frame_images(0).shape

	def image(self, index):
		i_width = 320
		i_height = 240
		path = self.video_folder + "/" + self.df['filename'][index]
		return img_to_array(load_img(full_path(path)))
		# return scipy.misc.imresize(original_image, (i_height, i_width))
