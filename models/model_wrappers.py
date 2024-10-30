from sklearn.ensemble import RandomForestClassifier
from aeon.classification.convolution_based import RocketClassifier
import numpy as np
from scipy.special import softmax


class RandomForest(RandomForestClassifier):

    def __init__(self,model):
        self.model = model
        self.trained = False

    def fit(self, X, y):
        X = np.reshape(X, ( X.shape[0] ,-1))
        self.model.fit(X, y)
        self.trained = True

    def predict_proba(self, X):
        X = np.reshape(X, ( X.shape[0] ,-1))
        return self.model.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        X = np.reshape(X, ( X.shape[0] ,-1))
        return self.model.score(X,y)

    def predict(self,X):
        X = np.reshape(X, ( X.shape[0] ,-1))
        return self.model.predict(X)




class MiniRocket(RocketClassifier):

    def __init__(self,model):
        self.model = model

        self.dataset_transform = None
        self.deicision_function = None
        self.binary_deicision_function = None
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.dataset_transform = (
            self.model.pipeline_[0].transform,
            self.model.pipeline_[1].transform
        )

        self.deicision_function = self.model.pipeline_[-1].decision_function
        self.binary_deicision_function = self.model.pipeline_[-1]._predict_proba_lr
        self.trained = True

    def predict(self,X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X,y)

    def predict_proba(self, X):

        X_transformed = self.dataset_transform[1](self.dataset_transform[0](X))

        if self.model.classes_.shape[0]>2:
            dists = self.deicision_function(X_transformed)
            probas = softmax(dists,axis=-1)
        else:
            probas = self.binary_deicision_function(X_transformed)

        return probas