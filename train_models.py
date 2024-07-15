import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from aeon.classification.convolution_based import RocketClassifier
from CNN_models import ResNetBaseline, TSDataset
from torch.utils.data import DataLoader
from utils import Trainer
from aeon.classification.interval_based import QUANTClassifier


def train_randomForest(X_train, y_train, X_test, y_test, dataset_name):
	X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

	clf = RandomForestClassifier(n_jobs=1)
	print("training random forest")
	clf.fit(X_train, y_train)
	acc = clf.score(X_test, y_test)
	preds = clf.predict(X_test)
	print("random forest accuracy is", acc)

	if dataset_name is not None:
		dump(clf, "trained_models/randomForest_" + dataset_name + ".bz2")

	return clf, preds


def train_miniRocket(X_train, y_train, X_test, y_test, dataset_name):

	clf = RocketClassifier(rocket_transform='miniRocket',n_jobs=20)
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
	clf = ResNetBaseline(in_channels=c_in, mid_channels=128, num_pred_classes=c_out).to(device)

	# get pytorch data loader
	batch_size = 32
	train_loader = DataLoader(TSDataset(X_train, y_train,device=device), batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(TSDataset(X_test, y_test, device=device), batch_size=batch_size, shuffle=False)

	# train resNet
	trainer = Trainer(model=clf)
	print("training ResNet")
	model_pah = "trained_models/resNet_" + dataset_name + ".pt" if dataset_name is not None else "trained_models/tmp_resNet"
	outs, acc = trainer.train(n_epochs=1000, train_loader=train_loader,model_path=model_pah,
	                          test_loader=test_loader, n_epochs_stop=100)

	# load best model from disk (early stopping)
	clf = torch.load(model_pah, map_location=device)

	# get predictions in batches
	preds = []
	for i in range(0,X_test.shape[0],batch_size):
		current_test = test_loader.dataset.samples[i: min(i+batch_size,X_test.shape[0])]
		preds.append( clf(current_test).detach().cpu().numpy() )
	preds = np.concatenate(preds)
	print("accuracy for resNet is ", acc.item())

	return clf, preds
