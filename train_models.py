import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from aeon.transformations.collection.convolution_based import MiniRocketMultivariate
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from CNN_models import ResNetBaseline, ModelCNN,TSDataset
from torch.cuda import is_available as is_GPU_available
from torch.utils.data import DataLoader

def train_randomForest(X_train,y_train, X_test, y_test,dataset_name):
	X_train, X_test = X_train.reshape(X_train.shape[0],-1), X_test.reshape(X_test.shape[0],-1)

	clf = RandomForestClassifier()
	print("training random forest")
	clf.fit(X_train, y_train)
	acc = clf.score(X_test,y_test)
	preds = clf.predict(X_test)
	print("random forest accuracy is",acc)

	if dataset_name is not None:
		dump ( clf, "trained_models/randomForest_"+dataset_name)

	return clf, preds

def train_miniRocket(X_train,y_train, X_test, y_test,dataset_name):
	clf = make_pipeline( MiniRocketMultivariate(n_jobs=-1),StandardScaler(),
			LogisticRegressionCV(max_iter=200, n_jobs=-1, ) )

	print ("training miniiRocket")
	clf.fit(X_train,y_train)
	score = clf.score(X_test,y_test)
	preds = clf.predict(X_test)
	print("accuracy for miniRocket is ", score)


	if dataset_name is not None:
		dump ( clf, "trained_models/randomForest_"+dataset_name)

	return clf, preds

def train_ResNet(X_train,y_train, X_test, y_test,dataset_name):
	device = device = "cuda" if is_GPU_available() else "cpu"
	print(device)

	c_in = X_train.shape[1]
	c_out = len(np.unique(y_train))

	resNet_arch = ResNetBaseline(in_channels=c_in,mid_channels=64,num_pred_classes=c_out).to(device)
	clf = ModelCNN( model=resNet_arch, n_epochs_stop=30, device=device, )

	train_loader = DataLoader(TSDataset(X_train,y_train),batch_size=64,shuffle=True)
	test_loader = DataLoader(TSDataset(X_test,y_test), batch_size=64, shuffle=False)

	print("training ResNet")

	acc = clf.train(num_epochs=100,train_loader=train_loader,test_loader=test_loader)
	print("accuracy for ResNet is",acc)

	return clf,acc