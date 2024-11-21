import numpy as np
import scipy
from utils import change_points_to_lengths

class Entropy_metric():

    def __init__(self, base="auto"):
        self.base = base

    def fit_data(self, X_train, X_test, y_train, y_test):
        self.n_test_samples, self.n_channels, self.n_timepoints = X_test.shape
        self.n_features = self.n_channels * self.n_timepoints

    def fit_ml_model(self, ml_model):
        pass

    def entropy(self, segments):

        #lengths = np.array(list(map(lambda x: change_points_to_lengths(y, X_train.shape[-1])), x)), segments))

        entropies = np.array(list(map(
			lambda segmentation: list(map(
				lambda channel_segments: scipy.stats.entropy(change_points_to_lengths(channel_segments, self.n_timepoints), base=len(channel_segments)) if len(channel_segments)>=2 else 1.0,
				segmentation)),
			segments)))

        entropy_mean = np.mean(entropies)
        entropy_std = np.std(entropies)

        return (("mean", entropy_mean), ("std", entropy_std),)

    def evaluate(self, segments):
        return self.entropy(segments)