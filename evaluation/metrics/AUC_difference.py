import numpy as np
from models.predictor_utils import predict_proba
from sklearn.metrics import auc as sklearn_auc

def normalize_saliency(saliency):
    return (np.abs(saliency) + 1e-5) / (np.max(np.abs(saliency)) + 1e-5)

def calc_AUC_score(A, A_pred, B, B_pred, start_points, end_points, salient_order, n_channels, n_timepoints, sample_class_pred_idx, classifier, mode="deletion"):

    if A_pred == B_pred:
        #print(f"Warning: original and replacement predictions are the same ({A_pred})")
        return 0.0, 0.0

    preds = [A_pred]
    perturbed_sample = A.copy()
    for start_idx, end_idx in zip(start_points, end_points):
        perturbed_idxs = salient_order[start_idx: end_idx]
        perturbed_sample[perturbed_idxs] = B[perturbed_idxs]
        perturbed_sample_reshaped = perturbed_sample.reshape(1, n_channels, n_timepoints)
        perturbed_sample_pred = predict_proba(classifier, perturbed_sample_reshaped)[0][sample_class_pred_idx]
        # except AttributeError:
        #     perturbed_sample_pred[unique_class]["pred"] = np.array(classifier.predict(perturbed_sample_reshaped)[0][sample_class_pred_idx], ndmin=1)

        preds.append(perturbed_sample_pred)
    preds.append(B_pred)
    if mode=="deletion":
        preds = (np.array(preds) - B_pred) #np.abs() # B pred will be opprosite class in the insertion case
    elif mode=="insertion":
        preds = (np.array(preds) - A_pred) #np.abs() # A pred will be opprosite class in the insertion case
    else:
        raise ValueError(f"mode must be deletion or insertion, but it is {mode}")

    AUC_score = sklearn_auc(x=np.linspace(0, 1, len(preds)), y=preds)
    normalized_AUC_score = AUC_score / np.abs(A_pred - B_pred)

    return AUC_score, normalized_AUC_score


def AUC_difference(classifier, X_train, X_test, y_train, attributions, y_test_pred, label_mapping, n_steps=18):

    n_samples, n_channels, n_timepoints = X_test.shape
    n_steps = np.min([n_steps, n_timepoints-2])

    classes = np.unique(y_train) #TODO: this is supposed to be label_mapping but label mapping currently doesn't have the class labels
    opposite_class_dict = {}
    for unique_class in classes:
        opposite_class_idxs = ~(y_train==unique_class)
        opposite_sample = np.mean(X_train[opposite_class_idxs], axis=0)
        opposite_class_dict[unique_class] = {"sample": opposite_sample, "pred": None}
        opposite_class_dict[unique_class]["pred"] = predict_proba(classifier, opposite_sample[None, :])[0]
        # print(opposite_class_dict[unique_class]["pred"])
        # except AttributeError:
        #     opposite_class_dict[unique_class]["pred"] = np.array(classifier.predict(opposite_sample[None, :])[0], ndmin=1)
        #     # resnet direct and to numpy

    AUC_diff_array = np.zeros(X_test.shape[0])
    normalized_AUC_diff_array = np.zeros(X_test.shape[0])

    for sample_idx, sample in enumerate(X_test):
        importance = normalize_saliency(attributions[sample_idx])
        salient_order = np.argsort(importance.reshape(n_channels * n_timepoints))[::-1] # decreasing
        change_points = np.array(np.round(np.linspace(0, n_channels * n_timepoints, n_steps + 2)), dtype=int)
        start_points = change_points[:-2]
        end_points = change_points[1:-1]

        # deletion
        sample_pred = y_test_pred[sample_idx] # TODO: this is the normal line of code, below a temporary until y_pred format is fixed
        # sample_pred = predict_proba(classifier, sample[None, :])[0]
        
        sample_class_pred_idx = np.argmax(sample_pred)
        sample_class = classes[sample_class_pred_idx]
        sample_class_pred = sample_pred[sample_class_pred_idx]
        opposite_sample = opposite_class_dict[sample_class]["sample"]
        opposite_class_pred = opposite_class_dict[sample_class]["pred"][sample_class_pred_idx]

        A = sample.reshape(n_channels * n_timepoints).copy()
        A_pred = sample_class_pred
        B = opposite_sample.reshape(n_channels * n_timepoints).copy()
        B_pred = opposite_class_pred
        AUC_deletion, normalized_AUC_deletion = calc_AUC_score(A, A_pred, B, B_pred, start_points, end_points, salient_order, n_channels, n_timepoints, sample_class_pred_idx, classifier, mode="deletion")

        A = opposite_sample.reshape(n_channels * n_timepoints).copy()
        A_pred = opposite_class_pred
        B = sample.reshape(n_channels * n_timepoints).copy()
        B_pred = sample_class_pred
        AUC_insertion, normalized_AUC_insertion = calc_AUC_score(A, A_pred, B, B_pred, start_points, end_points, salient_order, n_channels, n_timepoints, sample_class_pred_idx, classifier, mode="insertion")

        AUC_diff = AUC_insertion - AUC_deletion
        AUC_diff_array[sample_idx] = AUC_diff

        normalized_AUC_diff = normalized_AUC_insertion - normalized_AUC_deletion
        normalized_AUC_diff_array[sample_idx] = normalized_AUC_diff

    mean_AUC_diff = np.mean(AUC_diff_array)
    mean_normalized_AUC_diff = np.mean(normalized_AUC_diff_array)

    return ("default", mean_AUC_diff), ("normalized", mean_normalized_AUC_diff)