import numpy as np
from models.predictor_utils import predict_proba
from sklearn.metrics import auc as sklearn_auc
import copy

class AUIDC_metric():

    def __init__(self, max_steps=25):
        self.max_steps = max_steps

    def fit_data(self, X_train, X_test, y_train, y_test):
        self.n_test_samples, self.n_channels, self.n_timepoints = X_test.shape
        self.n_features = self.n_channels * self.n_timepoints
        self.n_steps = np.min([self.max_steps, self.n_features - 1])
        self.change_points = np.array(np.round(np.linspace(0, self.n_features, self.n_steps + 2)), dtype=int)[:-1] # includes 0 but not end point, each change point signifies the left closed bracket of the [) interval until the next change point non-inclusive
        self.X_test = X_test
        self.y_test = y_test

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


    def normalize_saliency(self, saliency):
        abs_saliency = np.abs(saliency) + 1e-6
        return abs_saliency / np.max(abs_saliency)

    def calc_AUC_score(self, A, A_pred, B, B_pred, salient_order, target_class, ml_model, mode="deletion"):
        # target_class = sample_class_pred_idx
        # A and B are flattened, and so is salient order

        preds = [A_pred]
        perturbed_sample = np.copy(A)
        replacement_sample = np.copy(B)
        start_points = self.change_points[:-1]
        end_points = self.change_points[1:]
        for start_idx, end_idx in zip(start_points, end_points): # very last segment is not computed since it will result in B
            perturbed_idxs = salient_order[start_idx: end_idx]
            perturbed_sample[perturbed_idxs] = replacement_sample[perturbed_idxs]
            perturbed_sample_reshaped = perturbed_sample.reshape(1, self.n_channels, self.n_timepoints)
            perturbed_sample_pred = predict_proba(ml_model, perturbed_sample_reshaped)[0][target_class]
            preds.append(perturbed_sample_pred)
        preds.append(B_pred)

        if mode=="deletion":
            preds = np.abs(np.array(preds) - B_pred) # B pred will be opposite class in the deletion case
        elif mode=="insertion":
            preds = np.abs(np.array(preds) - A_pred) # A pred will be opposite class in the insertion case
        else:
            raise ValueError(f"mode must be deletion or insertion, but it is {mode}")

        AUC_score = sklearn_auc(x=np.linspace(0, 1, len(preds)), y=preds) # len(preds) should be self.n_steps + 2

        return AUC_score


    def AUC_difference(self, attributions, y_test_pred, ml_model):

        AUC_insertion_array, AUC_deletion_array = np.zeros(self.n_test_samples), np.zeros(self.n_test_samples)

        for sample_idx, sample in enumerate(self.X_test):
            importance = self.normalize_saliency(attributions[sample_idx])
            salient_order = np.argsort(importance.reshape(self.n_features))[::-1] # decreasing

            # deletion
            sample_pred = y_test_pred[sample_idx] # TODO: this is the normal line of code, below a temporary until y_pred format is fixed
            # sample_pred = predict_proba(ml_model, sample[None, :])[0]

            predicted_class_idx = np.argmax(sample_pred)
            predicted_class = self.classes[predicted_class_idx]
            prediction_for_class = sample_pred[predicted_class_idx]
            opposite_class = self.opposite_class_dict[predicted_class]
            opposite_sample = self.mean_class_dict[opposite_class]["sample"]
            opposite_prediction_for_class = self.mean_class_dict[opposite_class]["pred"][predicted_class_idx]

            A = sample.reshape(self.n_features)
            A_pred = prediction_for_class
            B = opposite_sample.reshape(self.n_features)
            B_pred = opposite_prediction_for_class
            AUC_deletion = self.calc_AUC_score(A, A_pred, B, B_pred, salient_order, predicted_class_idx, ml_model, mode="deletion")

            A = opposite_sample.reshape(self.n_features)
            A_pred = opposite_prediction_for_class
            B = sample.reshape(self.n_features)
            B_pred = prediction_for_class
            AUC_insertion = self.calc_AUC_score(A, A_pred, B, B_pred, salient_order, predicted_class_idx, ml_model, mode="insertion")

            # AUC_diff = AUC_insertion - AUC_deletion
            AUC_insertion_array[sample_idx] = AUC_insertion
            AUC_deletion_array[sample_idx] = AUC_deletion

        mean_AUC_insertion = np.mean(AUC_insertion_array) # std
        std_AUC_insertion = np.std(AUC_insertion_array)
        mean_AUC_deletion = np.mean(AUC_deletion_array) # std
        std_AUC_deletion = np.std(AUC_deletion_array) # std

        return (("insertion_mean", mean_AUC_insertion), ("deletion_mean", mean_AUC_deletion), ("insertion_std", std_AUC_insertion), ("deletion_std", std_AUC_deletion)) #, ("normalized", mean_normalized_AUC_diff)

    def evaluate(self, attributions, y_test_pred, ml_model):
        return self.AUC_difference(attributions, y_test_pred, ml_model)