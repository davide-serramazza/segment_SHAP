import numpy as np
# import scipy
# from utils import change_points_to_lengths

class OrderChange_metric():

    def __init__(self, base="auto"):
        self.base = base

    def fit_data(self, X_train, X_test, y_train, y_test):
        self.n_test_samples, self.n_channels, self.n_timepoints = X_test.shape
        self.n_features = self.n_channels * self.n_timepoints

    def fit_ml_model(self, ml_model):
        pass

    def order_comparison(self, representatives_a, representatives_b):
        # representatives 1d vectors of same size
        order_check = not (np.argsort(representatives_a)==np.argsort(representatives_b)).all()

        return order_check

    def order_change(self, default_attributions, normalized_attributions, segments):

        #lengths = np.array(list(map(lambda x: change_points_to_lengths(y, X_train.shape[-1])), x)), segments))

        representatives_default = [np.concatenate([channel_att[channel_seg] for channel_att, channel_seg in zip(sample_att, sample_seg)]) for sample_att, sample_seg in zip(default_attributions, segments)] # shape = n_samples, n_segments_per_sample
        representatives_normalized = [np.concatenate([channel_att[channel_seg] for channel_att, channel_seg in zip(sample_att, sample_seg)]) for sample_att, sample_seg in zip(normalized_attributions, segments)]

        order_check = np.array([self.order_comparison(representatives_a, representatives_b) for representatives_a, representatives_b in zip(representatives_default, representatives_normalized)], dtype=bool) # shape=n_samples

        mean_order_change = order_check.mean(axis=0)


        return (("fraction", mean_order_change), )

    def evaluate(self, default_attributions, normalized_attributions, segments):
        return self.order_change(default_attributions, normalized_attributions, segments)