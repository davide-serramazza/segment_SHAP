import numpy as np
from aeon.segmentation._clasp import ClaSPSegmenter
import torch

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

def get_claSP_segmentation(X):
	result = []
	for ch in range(X.shape[0]):

		uts = X[ch]
		clasp = ClaSPSegmenter(n_cps=6, period_length=3)        # for gun point, actually best for both!
		found_cps, profiles, scores = clasp._run_clasp(uts)

		# it seems that they are not sorted
		found_cps.sort()
		if found_cps[0]!=0:
			found_cps = np.append(0,found_cps)

		result.append( found_cps )

	result.append( [] )
	return  np.array(result, dtype=object)
