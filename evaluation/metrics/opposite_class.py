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

        self.classes = np.unique(y_train) #TODO: this is supposed to be label_mapping but label mapping currently doesn't have the class labels
        self.mean_class_dict = dict.fromkeys(self.classes)
        self.opposite_class_dict = dict.fromkeys(self.classes)

        for unique_class in self.classes:
            class_idxs = (y_train==unique_class)
            mean_class_sample = np.mean(X_train[class_idxs], axis=0)
            self.mean_class_dict[unique_class] = {"sample": mean_class_sample, "pred": None}
            # print(opposite_class_dict[unique_class]["pred"])

    def fit_ml_model(self, ml_model):

        for unique_class in self.classes:
            mean_class_sample = self.mean_class_dict[unique_class]["sample"]
            mean_class_pred = predict_proba(ml_model, mean_class_sample[None, :])[0]
            self.mean_class_dict[unique_class]["pred"] = mean_class_pred

        for target_class in self.classes:
            mean_class_preds_for_target_class = {unique_class: self.mean_class_dict[unique_class]["pred"][target_class] for unique_class in self.classes} # {class: prediction of avg_class_sample for target_class}
            opposite_class = min(mean_class_preds_for_target_class, key=mean_class_preds_for_target_class.get)
            self.opposite_class_dict[target_class] = opposite_class

    def evaluate(self):
        preds = []
        for i, target_class in enumerate(self.classes):
            opposite_class = self.opposite_class_dict[target_class]
            preds.append(self.mean_class_dict[opposite_class]["pred"][i])

        opp_class_pred_mean = np.mean(preds)
        opp_class_pred_std = np.std(preds)


        return (("mean", opp_class_pred_mean), ("std", opp_class_pred_std),)