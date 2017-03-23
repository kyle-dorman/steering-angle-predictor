

def train_bottleneck_features(dataset, batch_size):
	if dataset == 'original':
		data = SimulatorData(load_simple_data(), batch_size)
	else:
		raise Exception("Unexpected dataset:", dataset)

	train_output_file, validation_output_file = files(dataset, batch_size)

	print("Saving to ...")
	print(train_output_file)
	print(validation_output_file)

	model = VGG16(input_tensor=Input(shape=data.feature_shape), pooling=None, include_top=False)

	print('Bottleneck training')
	bottleneck_features_train = model.predict_generator(data.train_generator(), data.num_train)
	pickle_data = { 'features': bottleneck_features_train, 'labels': data.train_labels() }
	pickle.dump(pickle_data, open(train_output_file, 'wb'))

	print('Bottleneck validation')
	bottleneck_features_validation = model.predict_generator(data.validation_generator(), data.num_validation)
	pickle_data = { 'features': bottleneck_features_validation, 'labels': data.validation_labels() }
	pickle.dump(pickle_data, open(validation_output_file, 'wb'))