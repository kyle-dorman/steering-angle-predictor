#!/bin/python

from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Reshape, Dropout
from keras.models import Model
from keras.layers.recurrent import GRU

from util import download_s3
from bottleneck_generator import BottleneckData

def model(load_saved=False):
	if load_saved:
		print("load_saved")
	else:
		create_model()

#image_inputs = Input(shape=(3, 7, 10, 512), dtype='float32', name='bottleneck_left_right_center')
def create_model(image_inputs, batch_size=32, video_frames=100):
	
	image_x = Reshape((video_frames, -1))(image_inputs)
	vehicle_inputs = Input(batch_shape=(batch_size, video_frames, 3), dtype='float32', name='angle_torque_speed')
	x = layers.merge([image_x, vehicle_inputs], mode='concat', concat_axis=2)
	x = GRU(256,activation='relu', 
		dropout_W=0.2, dropout_U=0.2, stateful=True)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(32, activation='relu')(x)
	x = Dropout(0.2)(x)
	output = Dense(1, name='output')(x)

	model = Model(input=[image_inputs, vehicle_inputs], output=output)
	model.compile(optimizer='adam', loss='mean_squared_error')

	return model

def train_model(model, epochs=1, batch_size=32, video_frames=100):
	data = BottleneckData(batch_size=batch_size, video_frames=video_frames)

	for i in range(epochs):
		for dataset in data.datasets:
			model.reset_states()
			model.fit_generator(dataset.train_generators[0], 
				dataset.train_generators[0].size(), 
				nb_epoch=1, 
				verbose=1, 
				validation_data=dataset.valid_generators[0], 
				nb_val_samples=dataset.valid_generators[0].size())

def download_bottleneck_features():
	for i in [1,2,4,5,6]:
		download_s3("bottleneck_data/HMB_{}.p".format(i))

video_frames = 50
batch_size = 1
rnn_model = create_model(Input(batch_shape=(batch_size, video_frames, 3, 7, 10, 512), dtype='float32', name='bottleneck_left_right_center'), 
	batch_size=batch_size, video_frames=video_frames)
train_model(rnn_model, video_frames=video_frames)
