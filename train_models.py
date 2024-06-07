import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump, load
from aeon.transformations.collection.convolution_based import MiniRocketMultivariate
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from CNN_models import ResNetBaseline, TSDataset
from torch.cuda import is_available as is_GPU_available
from torch.utils.data import DataLoader
from utils import Trainer
from aeon.classification.interval_based import QUANTClassifier


def train_randomForest(X_train, y_train, X_test, y_test, dataset_name):
	X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

	clf = RandomForestClassifier(n_jobs=-1)
	print("training random forest")
	clf.fit(X_train, y_train)
	acc = clf.score(X_test, y_test)
	preds = clf.predict(X_test)
	print("random forest accuracy is", acc)

	if dataset_name is not None:
		dump(clf, "trained_models/randomForest_" + dataset_name + ".bz2")

	return clf, preds


def train_miniRocket(X_train, y_train, X_test, y_test, dataset_name):
	clf = make_pipeline(MiniRocketMultivariate(n_jobs=-1), StandardScaler(),
	                    LogisticRegressionCV(max_iter=200, n_jobs=-1, ))

	print("training miniRocket")
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	preds = clf.predict(X_test)
	print("accuracy for miniRocket is ", score)

	if dataset_name is not None:
		dump(clf, "trained_models/miniRocket_" + dataset_name + ".bz2")

	return clf, preds


def train_QUANT(X_train, y_train, X_test, y_test, dataset_name):
	clf = QUANTClassifier()

	print("training QUANT")
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	preds = clf.predict(X_test)
	print("accuracy for QUANT is ", score)

	if dataset_name is not None:
		dump(clf, "trained_models/QUANT_" + dataset_name + ".bz2")

	return clf, preds


def train_ResNet(X_train, y_train, X_test, y_test, dataset_name, device):
	# get  number of in channel (c_in) , last layer output (c_out)
	c_in = X_train.shape[1]
	c_out = len(np.unique(y_train))

	# instantiate ResNet
	clf = ResNetBaseline(in_channels=c_in, mid_channels=64, num_pred_classes=c_out).to(device)

	# get pytorch data loader
	train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=32, shuffle=True)
	test_loader = DataLoader(TSDataset(X_test, y_test), batch_size=32, shuffle=False)
	trainer = Trainer(model=clf)

	print("training ResNet")
	outs, acc = trainer.train(n_epochs=100, train_loader=train_loader, test_loader=test_loader, n_epochs_stop=30)
	preds = torch.argmax(outs, dim=-1).detach().cpu().numpy()
	print("accuracy for resNet is ", acc.item())

	if dataset_name is not None:
		torch.save(clf, "trained_models/resNet_" + dataset_name + ".pt")

	return clf, preds
