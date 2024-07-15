import numpy as np
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_gunpoint as load_gunpoint
from aeon.datasets import load_from_tsfile
import pandas as pd

def load_data(subset, dataset_name):
	if subset not in {'train', 'test', 'all'}:
		raise ValueError(
			"subset has to be  either 'train' ,'test' or 'all'"
		)

	#TODO to be reintroduced
	#if dataset_name not in {'UWAVE', 'gunpoint', 'synth_uni', 'synth_fixed_multi'}:
#		raise ValueError(
	#		"dataset_name note recognized"
#		)

	if dataset_name == "UWAVE":
		X_train, y_train = np.load("datasets/UWAVE/Xtr.npy"), np.load("datasets/UWAVE/Ytr.npy")
		X_test, y_test = np.load("datasets/UWAVE/Xte.npy"), np.load("datasets/UWAVE/Yte.npy")

		# channels as second dimension
		X_train, X_test = (np.transpose(X_train, (0, 2, 1)).astype(np.float32),
		                   np.transpose(X_test, (0, 2, 1)).astype(np.float32))
		y_train, y_test = np.reshape(y_train, y_train.shape[0]), np.reshape(y_test, y_test.shape[0])

	elif dataset_name == "gunpoint":
		X_train, y_train = load_gunpoint(split="train")
		X_test, y_test = load_gunpoint(split="test")
		X_train = X_train.astype(np.float32) ;  X_test = X_test.astype(np.float32)

	elif dataset_name.startswith("synth"):
		X_train, X_test, y_train, y_test= load_synth_data(dataset_name)

	elif dataset_name == "MP50":
		X_train, y_train = load_from_tsfile("datasets/MilitaryPress/TRAIN_full_X.ts")
		X_test, y_test = load_from_tsfile("datasets/MilitaryPress/TEST_full_X.ts")
		X_train ,X_test = X_train.astype(np.float32), X_test.astype(np.float32)

	elif dataset_name == "MP8":
		data = np.load("datasets/MilitaryPress/MP_centered.npy", allow_pickle=True).item()
		X_train, y_train = data["train"]["X"].astype(np.float32), data["train"]["y"]
		X_test, y_test = data["test"]["X"].astype(np.float32), data["test"]["y"]

	elif dataset_name == "EOG":
		# load horizontal signal
		X_train_h, y_train_h = load_from_tsfile("datasets/EOGSignal/EOGHorizontalSignal_TRAIN.ts")
		X_test_h, y_test_h = load_from_tsfile("datasets/EOGSignal/EOGHorizontalSignal_TEST.ts")

		# load vertical signal
		X_train_v, y_train_v = load_from_tsfile("datasets/EOGSignal/EOGVerticalSignal_TRAIN.ts")
		X_test_v, y_test_v = load_from_tsfile("datasets/EOGSignal/EOGVerticalSignal_TEST.ts")

		# concatenate and convert to float32
		X_train = np.concatenate((X_train_h, X_train_v), axis=1)
		X_test = np.concatenate((X_test_h, X_test_v), axis=1)
		X_train ,X_test = X_train.astype(np.float32), X_test.astype(np.float32)

		# finally take labels
		assert np.all(y_train_h == y_train_v)
		assert np.all(y_test_h == y_test_v)
		y_train, y_test = y_train_h, y_test_h

	le = LabelEncoder()
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	if subset == 'train':
		return X_train, y_train
	elif subset == 'test':
		return X_test, y_test
	elif subset == 'all':
		return X_train, X_test, y_train, y_test, le.classes_


def load_synth_data(dataset_name):
	# TODO hard coded
	# select file
	if dataset_name.startswith("synth_fixed_multi"):
		data = np.load("/home/davide/Desktop/datasets/fixed_length.npy", allow_pickle=True).item()
	elif dataset_name.startswith("synth_varying_multi"):
		data = np.load("/home/davide/Desktop/datasets/varying_length.npy", allow_pickle=True).item()


	# select classification or regression target
	X_train, X_test = data['train']['X'], data['test']['X']
	if dataset_name.endswith("clf"):
		y_train, y_test = data['train']['y_clf'], data['test']['y_clf']
	elif dataset_name.endswith("reg"):
		# TODO int or float for regression task???
		y_train, y_test = data['train']['y_reg'].astype(float), data['test']['y_reg'].astype(float)

	return X_train, X_test, y_train, y_test

				                                                                   
"""
def load_synth_data(name):
	if name == "synth_uni":
		n_samples = 15000
		n_features = 1
		time_points = 500
		file_names = ['uni_signals.csv','uni_classes.csv']
	elif name == "synth_multi":
		n_samples = 15000
		n_features = 6
		time_points = 500
		file_names = ['multi_signals.csv','multi_classes.csv']

	df_signal = pd.read_csv('/home/davide/Downloads/dataset/' + file_names[0])
	df_class = pd.read_csv('/home/davide/Downloads/dataset/' + file_names[1])

	X = np.zeros(shape=(n_samples, n_features, time_points))
	for i in range(n_samples):
		for j in range(n_features):
			col_name = "sample_" + str(i) + "_feature" + str(j)
			X[i, j] = df_signal[col_name].values
	y = df_class['classes'].values
	X_train, y_train = X[:14500], y[:14500]
	X_test, y_test = X[14500:], y[14500:]
	return X_test, X_train, y_test, y_train
"""