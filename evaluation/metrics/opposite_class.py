import numpy as np
from models.predictor_utils import predict_proba
from sklearn.metrics import auc as sklearn_auc
import copy

class Opposite_Class():

    def __init__(self):
        pass

    def fit_data(self, X_train, X_test, y_train, y_test):
        # self.n_test_samples, self.n_channels, self.n_timepoints = X_test.shape
        # self.n_features = self.n_channels * self.n_timepoints
        # self.X_test = X_test
        # self.y_test = y_test

        self.classes = np.unique(y_train)
        self.opposite_class_dict = {}
        for unique_class in self.classes:
            opposite_class_idxs = ~(y_train==unique_class)
            opposite_sample = np.mean(X_train[opposite_class_idxs], axis=0)
            self.opposite_class_dict[unique_class] = {"sample": opposite_sample, "pred": None}
            # print(opposite_class_dict[unique_class]["pred"])

    def fit_ml_model(self, ml_model):

        for unique_class in self.classes:
            opposite_sample = self.opposite_class_dict[unique_class]["sample"]
            self.opposite_class_dict[unique_class]["pred"] = predict_proba(ml_model, opposite_sample[None, :])[0]
            # except AttributeError:
            #     opposite_class_dict[unique_class]["pred"] = np.array(ml_model.predict(opposite_sample[None, :])[0], ndmin=1)

    def evaluate(self):
        preds = []
        for i, opposite_class in enumerate(self.classes):
            preds.append(self.opposite_class_dict[opposite_class]["pred"][i])

        opp_class_pred_mean = np.mean(preds)
        opp_class_pred_std = np.std(preds)


        return (("mean", opp_class_pred_mean), ("std", opp_class_pred_std),)