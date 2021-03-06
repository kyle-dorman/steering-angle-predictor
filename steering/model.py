#!/bin/python

from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import History, ModelCheckpoint, Callback, EarlyStopping, CSVLogger, TensorBoard
import numpy as np
import scipy

from steering.util import download_s3, full_path
from steering.bottleneck_generator import BottleneckData

def get_image_processor_model(image_input):
	model = VGG16(input_tensor=image_input, include_top=False)
	output = BatchNormalization()(model.output)
	return Model(input=image_input, output=output)

def process_images(image_processor_model, images, batch_size, i_width, i_height):
	resized_images = np.array([scipy.misc.imresize(image, (i_height, i_width)) for image in images], dtype=np.float32)
	preprocess_images = preprocess_input(resized_images)
	return image_processor_model.predict(preprocess_images, batch_size=3)

def predict(reccurent_model, features):
	image_inputs = features[0]
	vehicle_data_inputs = features[1]
	# image_inputs = [np.zeros(image_input.shape()) for i in range(batch_size - 1)]
	# image_inputs.append(image_input)
	# vehicle_data_inputs = [np.zeros(vehicle_data.shape()) for i in range(batch_size - 1)]
	# vehicle_data_inputs.append(vehicle_data_inputs)

	return reccurent_model.predict({'bottleneck_left_right_center': image_inputs, 'angle_torque_speed': vehicle_data_inputs})

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

def train_model(model, data, config, include_tensorboard):
	model_history = History()
	model_history.on_train_begin()
	saver = ModelCheckpoint(full_path(config.model_file()), verbose=1, save_best_only=True, period=1)
	saver.set_model(model)
	early_stopping = EarlyStopping(min_delta=config.min_delta, patience=config.patience, verbose=1)
	early_stopping.set_model(model)
	early_stopping.on_train_begin()
	csv_logger = CSVLogger(full_path(config.csv_log_file()))
	csv_logger.on_train_begin()
	if include_tensorboard:
		tensorborad = TensorBoard(histogram_freq=10, write_images=True)
		tensorborad.set_model(model)
	else:
	 tensorborad = Callback()

	epoch = 0
	stop = False
	while(epoch <= config.max_epochs and stop == False):
		epoch_history = History()
		epoch_history.on_train_begin()
		valid_sizes = []
		train_sizes = []
		print("Epoch:", epoch)
		for dataset in data.datasets:
			print("dataset:", dataset.name)
			model.reset_states()
			dataset.reset_generators()

			valid_sizes.append(dataset.valid_generators[0].size())
			train_sizes.append(dataset.train_generators[0].size())
			fit_history = model.fit_generator(dataset.train_generators[0],
				dataset.train_generators[0].size(), 
				nb_epoch=1, 
				verbose=0, 
				validation_data=dataset.valid_generators[0], 
				nb_val_samples=dataset.valid_generators[0].size())

			epoch_history.on_epoch_end(epoch, last_logs(fit_history))

			train_sizes.append(dataset.train_generators[1].size())
			fit_history = model.fit_generator(dataset.train_generators[1],
				dataset.train_generators[1].size(),
				nb_epoch=1, 
				verbose=0)

			epoch_history.on_epoch_end(epoch, last_logs(fit_history))

		epoch_logs = average_logs(epoch_history, train_sizes, valid_sizes)
		model_history.on_epoch_end(epoch, logs=epoch_logs)
		saver.on_epoch_end(epoch, logs=epoch_logs)
		early_stopping.on_epoch_end(epoch, epoch_logs)
		csv_logger.on_epoch_end(epoch, epoch_logs)
		tensorborad.on_epoch_end(epoch, epoch_logs)
		epoch+= 1

		if early_stopping.stopped_epoch > 0:
			stop = True

	early_stopping.on_train_end()
	csv_logger.on_train_end()
	tensorborad.on_train_end({})

def average_logs(history, train_sizes, valid_sizes):
	valid_sample_size = sum(valid_sizes)
	train_sample_size = sum(train_sizes)
	logs = {}

	for k, v in history.history.items():
		if "val_" in k:
			logs[k] = sum([val*(valid_sizes[i]/valid_sample_size) for i, val in enumerate(v)])
		else:
			logs[k] = sum([val*(train_sizes[i]/train_sample_size) for i, val in enumerate(v)])
	return logs

def last_logs(history):
	return {k: v[-1] for k, v in history.history.items()}
