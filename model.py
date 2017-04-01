#!/bin/python

from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Reshape, Dropout
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.callbacks import History, ModelCheckpoint, Callback

from util import download_s3, full_path
from bottleneck_generator import BottleneckData

def model(load_saved=False):
	if load_saved:
		print("load_saved")
	else:
		create_model()

def create_model(image_inputs, vehicle_inputs, video_frames=100):
	image_x = Reshape((video_frames, -1))(image_inputs)
	x = layers.merge([image_x, vehicle_inputs], mode='concat', concat_axis=2)
	x = GRU(256,activation='relu', dropout_W=0.2, dropout_U=0.2, stateful=True)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(32, activation='relu')(x)
	x = Dropout(0.2)(x)
	output = Dense(1, name='output')(x)

	model = Model(input=[image_inputs, vehicle_inputs], output=output)
	model.compile(optimizer='adam', loss='mean_squared_error')

	return model

def train_model(model, data, epochs=1, batch_size=32, video_frames=100):
	all_history = HistoryMultiplexer()
	all_history.on_train_begin()
	saver = ModelCheckpoint(full_path("model.cptk"), verbose=1, save_best_only=True, period=1)

	for epoch in range(epochs):
		epoch_history = History()
		epoch_history.on_train_begin()
		valid_sizes = []
		train_sizes = []
		for dataset in data.datasets:
			model.reset_states()
			valid_sizes.append(dataset.valid_generators[0].size())
			train_sizes.append(dataset.train_generators[0].size())

			history = model.fit_generator(dataset.train_generators[0],
				dataset.train_generators[0].size(), 
				nb_epoch=1, 
				verbose=1, 
				validation_data=dataset.valid_generators[0], 
				nb_val_samples=dataset.valid_generators[0].size())

			epoch_history.on_epoch_end(epoch, history.history)

			train_sizes.append(dataset.train_generators[1].size())
			history = model.fit_generator(dataset.train_generators[1],
				dataset.train_generators[1].size(),
				nb_epoch=1, 
				verbose=1)

			epoch_history.on_epoch_end(epoch, history.history)

		all_history.on_epoch_end(epoch, epoch_history, train_sizes, valid_sizes)
		logs = {k: v[-1] for k, v in all_history.history.items()}
		print([k for k in logs.keys()])
		saver.on_epoch_end(epoch, logs=logs)

class HistoryMultiplexer(Callback):
	def on_train_begin(self, logs=None):
		self.history = {}
		self.epoch = []

	def on_epoch_end(self, epoch, history, train_sizes, valid_sizes):
		self.epoch.append(epoch)
		logs = self.logs(history, train_sizes, valid_sizes)
		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

	def logs(self, history, train_sizes, valid_sizes):
		valid_size = sum(valid_sizes)
		train_size = sum(train_sizes)
		logs = {}
		print('history:', history.history)
		for k, v in history.history.items():
			if "val_" in k:
				logs[k] = sum([val*(valid_sizes[i]/valid_size) for i, val in enumerate(v)])
			else:
				logs[k] = sum([val*(train_sizes[i]/train_size) for i, val in enumerate(v)])
		return logs

def download_bottleneck_features():
	for i in [1,2,4,5,6]:
		download_s3("bottleneck_data/HMB_{}.p".format(i))
