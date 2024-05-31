from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

def train_randomForest(X_train,y_train, X_test, y_test,dataset_name):
	X_train, X_test = X_train.reshape(X_train.shape[0],-1), X_test.reshape(X_test.shape[0],-1)

	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	acc = clf.score(X_test,y_test)
	preds = clf.predict(X_test)
	print("random forest accuracy is",acc)

	if dataset_name is not None:
		dump ( clf, "trained_models/randomForest_"+dataset_name)

	return clf, preds