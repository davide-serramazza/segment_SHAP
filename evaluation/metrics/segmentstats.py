import numpy as np
import scipy
from utils import change_points_to_lengths

class SegmentStats_metric():

    def __init__(self, base="auto"):
        self.base = base

    def fit_data(self, X_train, X_test, y_train, y_test):
        self.n_test_samples, self.n_channels, self.n_timepoints = X_test.shape
        self.n_features = self.n_channels * self.n_timepoints

    def fit_ml_model(self, ml_model):
        pass

    def segmentstats(self, segments):

        #lengths = np.array(list(map(lambda x: change_points_to_lengths(y, X_train.shape[-1])), x)), segments))

        lengths = [[change_points_to_lengths(channel_segments, self.n_timepoints) for channel_segments in instance_segments] for instance_segments in segments]


        entropies = np.array([[scipy.stats.entropy(channel_lengths, base=len(channel_lengths)) if len(channel_lengths)>=2 else 1.0 for channel_lengths in instance_lengths] for instance_lengths in lengths])

        entropy_mean = np.mean(entropies)
        entropy_std = np.std(entropies)

        n_segments = np.array([[len(channel_lengths) for channel_lengths in instance_lengths] for instance_lengths in lengths])
        max_segments = np.max(n_segments)
        n_segments_mean = np.mean(n_segments) #/ max_segments
        n_segments_std = np.std(n_segments) #/ max_segments
        # if min_segments < max_segments:
        #     lesser_segments = n_segments[n_segments < max_segments]
        #     percent_lesser_segments = lesser_segments.size / n_segments.size
        #     mean_lesser_segments = np.mean(lesser_segments)
        #     percent_mean_lesser_segments = mean_lesser_segments / max_segments
        # else:
        #     percent_lesser_segments = 0.0
        #     percent_mean_lesser_segments = 1.0


        return (("entropy_mean", entropy_mean), ("entropy_std", entropy_std), ("n_segments_mean", n_segments_mean), ("n_segments_std", n_segments_std))

    def evaluate(self, segments):
        return self.segmentstats(segments)