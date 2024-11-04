import numpy as np
from aeon.segmentation._clasp import ClaSPSegmenter
import torch
from LIMESegment.Utils.explanations import NNSegment
from sktime.annotation.igts import InformationGainSegmentation
from sklearn.preprocessing import MinMaxScaler 
from sktime.annotation.ggs import GreedyGaussianSegmentation
from sklearn.preprocessing import StandardScaler

def get_feature_mask(segments,series_length):
	feature_mask = torch.zeros((1,len(segments),series_length ))
	n_segment = 0
	for n_ch, ch_segments in enumerate(segments):

		current_segments = ch_segments.tolist()
		current_segments.append(series_length)
		for i in range( len(current_segments) - 1):
			start = current_segments[i]
			end = current_segments[i+1]
			feature_mask[0,n_ch, start:end ] = n_segment
			n_segment+=1

	assert  (feature_mask<0).sum() == 0

	return feature_mask

def get_claSP_segmentation(X , n_segments=5):

    n_change_points = n_segments - 1

    result = []
    clasp = ClaSPSegmenter(n_cps=n_segments, period_length=4)

    for ch in range(X.shape[0]):

        uts = X[ch]
        found_cps, profiles, scores = clasp._run_clasp(uts)

        found_cps.sort()
        if found_cps[0]!=0:
            found_cps = np.append(0,found_cps)

        result.append( found_cps )

    return np.fromiter(result, dtype=object)


def ensure_begins_with_0(L):
    if len(L)==0:
        L = [0]
    elif L[0] != 0:
        L = [0] + L
    return L

def get_NNSegment_segmentation(X, window_size=None, n_segments=5, **kwargs):
    n_change_points = n_segments - 1
    X = X.astype(np.float64)
    n_channels, n_timepoints = X.shape
    if window_size is None:
        window_size = n_timepoints // 5 # NNSegment default
    change_points_per_channel = [ensure_begins_with_0(NNSegment(channel, window_size=window_size, change_points=n_change_points, **kwargs)) for channel in X]
    change_points_per_channel = np.fromiter([np.array(current_seg) for current_seg in change_points_per_channel], dtype=object)
    return change_points_per_channel

def get_equal_segmentation(X, n_segments=5):
    n_channels, n_timepoints = X.shape
    segment_length = n_timepoints / n_segments
    change_points = np.array(np.round(np.arange(n_segments) * segment_length), dtype=int)
    change_points_per_channel = np.tile(change_points, (n_channels, 1))
    change_points_per_channel = np.fromiter(change_points_per_channel, dtype=object)
    return change_points_per_channel

def labels_to_changepoints(labels):
    labels = np.array(labels, dtype=int)
    keys = (labels[:-1] != labels[1:])
    change_points = np.append(0, np.arange(1, len(labels))[keys])
    return change_points

def get_InformationGain_segmentation(X, n_segments=5, step=5):
    n_change_points = n_segments - 1
    n_channels, n_timepoints = X.shape
    igts = InformationGainSegmentation(k_max=n_change_points, step=step)
    X = X.T
    X_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    if n_channels==1: # remedy univariate issue
        X_scaled = np.append(X_scaled, X_scaled.max() - X, axis=1)

    try:
        segmentation_labels = igts.fit_predict(X_scaled)
    except UnboundLocalError:
        segmentation_labels = np.array( [0] ,  dtype=object)

    change_points = labels_to_changepoints(segmentation_labels)
    change_points_per_channel = np.tile(change_points, (n_channels, 1))
    change_points_per_channel = np.fromiter(change_points_per_channel, dtype=object)
    return change_points_per_channel

def get_GreedyGaussian_segmentation(X, n_segments=5, **kwargs):
    n_change_points = n_segments - 1
    n_channels, n_timepoints = X.shape
    ggs = GreedyGaussianSegmentation(k_max=n_change_points, **kwargs) 
    X = X.T
    X_scaled = StandardScaler().fit_transform(X) 
    segmentation_labels = ggs.fit_predict(X_scaled) 
    change_points = labels_to_changepoints(segmentation_labels)
    change_points_per_channel = np.tile(change_points, (n_channels, 1))
    change_points_per_channel = np.fromiter(change_points_per_channel, dtype=object)
    return change_points_per_channel