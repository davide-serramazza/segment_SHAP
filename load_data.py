import numpy as np
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_gunpoint as load_gunpoint
from aeon.datasets import load_from_tsfile
import pandas as pd
import os

def load_data(subset, dataset_name, path="datasets"):
	if subset not in {'train', 'test', 'all'}:
		raise ValueError(
			"subset has to be  either 'train' ,'test' or 'all'"
		)

	if dataset_name == "UWAVE":
		X_train, y_train = np.load(os.path.join(path, "UWAVE/Xtr.npy")), np.load(os.path.join(path, "UWAVE/Ytr.npy"))
		X_test, y_test = np.load(os.path.join(path, "UWAVE/Xte.npy")), np.load(os.path.join(path, "UWAVE/Yte.npy"))

		# channels as second dimension
		X_train, X_test = (np.transpose(X_train, (0, 2, 1)).astype(np.float32),
		                   np.transpose(X_test, (0, 2, 1)).astype(np.float32))
		y_train, y_test = np.reshape(y_train, y_train.shape[0]), np.reshape(y_test, y_test.shape[0])

	elif dataset_name == "gunpoint":
		X_train, y_train = load_gunpoint(split="train")
		X_test, y_test = load_gunpoint(split="test")
		X_train = X_train.astype(np.float32) ;  X_test = X_test.astype(np.float32)

	elif dataset_name == "MP50":
		X_train, y_train = load_from_tsfile(os.path.join(path, "MilitaryPress/TRAIN_full_X.ts"))
		X_test, y_test = load_from_tsfile(os.path.join(path, "MilitaryPress/TEST_full_X.ts"))
		X_train ,X_test = X_train.astype(np.float32), X_test.astype(np.float32)

	elif dataset_name == "MP8":
		data = np.load(os.path.join(path, "MilitaryPress/MP_centered.npy"), allow_pickle=True).item()
		X_train, y_train = data["train"]["X"].astype(np.float32), data["train"]["y"]
		X_test, y_test = data["test"]["X"].astype(np.float32), data["test"]["y"]

	elif dataset_name == "EOG":
		# load horizontal signal
		X_train_h, y_train_h = load_from_tsfile(os.path.join(path, "EOGSignal/EOGHorizontalSignal_TRAIN.ts"))
		X_test_h, y_test_h = load_from_tsfile(os.path.join(path, "EOGSignal/EOGHorizontalSignal_TEST.ts"))

		# load vertical signal
		X_train_v, y_train_v = load_from_tsfile(os.path.join(path, "EOGSignal/EOGVerticalSignal_TRAIN.ts"))
		X_test_v, y_test_v = load_from_tsfile(os.path.join(path, "EOGSignal/EOGVerticalSignal_TEST.ts"))

		# concatenate and convert to float32
		X_train = np.concatenate((X_train_h, X_train_v), axis=1)
		X_test = np.concatenate((X_test_h, X_test_v), axis=1)
		X_train ,X_test = X_train.astype(np.float32), X_test.astype(np.float32)

		# finally take labels
		assert np.all(y_train_h == y_train_v)
		assert np.all(y_test_h == y_test_v)
		y_train, y_test = y_train_h, y_test_h

	elif dataset_name == "KeplerLightCurves":
		X_train , y_train = load_from_tsfile( os.path.join(path, "KeplerLightCurves/KeplerLightCurves_TRAIN.ts") )
		X_test , y_test = load_from_tsfile( os.path.join(path, "KeplerLightCurves/KeplerLightCurves_TEST.ts") )
		X_train ,X_test = X_train.astype(np.float32), X_test.astype(np.float32)

	elif dataset_name.startswith("synth"):
		X_train, X_test, y_train, y_test = load_synth_data(dataset_name,path)

	else:
		raise ValueError(f"Dataset {dataset_name} is not recognized")

	le = LabelEncoder()
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	if subset == 'train':
		return X_train, y_train
	elif subset == 'test':
		return X_test, y_test
	elif subset == 'all':
		return X_train, X_test, y_train, y_test, le.classes_


def load_synth_data(dataset_name, path):
	# TODO hard coded
	# select file
	synth_path = os.path.join(path,"synth_data")
	if dataset_name.startswith("synth_fixedLength"):
		data = np.load( os.path.join(synth_path, 'fixed_length.npy'), allow_pickle=True).item()
	elif dataset_name.startswith("synth_oneWave"):
		data = np.load(  os.path.join(synth_path, 'one_wave.npy'), allow_pickle=True).item()
	else:
		raise ValueError(f"Synthetic dataset {dataset_name} is not recognized,it has to be either \
			fixedLength or oneWave")

	# select classification or regression target

	X_train, X_test = data['train']['X'], data['test']['X']
	if dataset_name.endswith("clf"):
		y_train, y_test = data['train']['y_clf'], data['test']['y_clf']
	elif dataset_name.endswith("reg"):
		y_train, y_test = data['train']['y_reg'].astype(np.float32), data['test']['y_reg'].astype(np.float32)
	else:
		raise ValueError(f"Synthetic dataset {dataset_name} is not recognized, it has to be either clf or reg")

	return X_train, X_test, y_train, y_test