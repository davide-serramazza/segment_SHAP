import numpy as np
import torch
import pandas as pd
from copy import deepcopy

def forward_classification(X_test: torch.Tensor, model):

	# convert X to pytorch tensor
	X_test_numpy = X_test.detach().numpy()
	# compute probability
	predictions = model.predict_proba(X_test_numpy)
	# return result as torch tensor as expected by captum attribution method
	return torch.tensor(predictions)


def sample_background(X_train, n):
	to_select = np.random.permutation(X_train.shape[0])[:n]
	return torch.tensor(X_train[to_select])


def extract_InterpretTime_info(X_test, X_train, dataset_name, y_test, y_train):
	n_channels, seq_len = X_train.shape[1:]
	n_classes = len(np.unique(y_train))
	# create a dictionary storing the dataset and metadata as name , local/global mean/std for
	test_set_dict = {'name': dataset_name,  # name

	                 "X": X_test, "y": y_test, "train_y": y_train,  # splits

	                 "seq_len": seq_len, "n_channels": n_channels, "n_classes": n_classes,  # TS infos

	                 "local_mean": pd.DataFrame(np.mean(X_train, axis=0)),  # distributions to be used
	                 "local_std": pd.DataFrame(np.std(X_train, axis=0)),
	                 "global_mean": np.mean(X_train),
	                 "global_std": np.std(X_train)
	                 }

	return test_set_dict

def intantiate_dict_results(explanations, masks):
	def clean(d):
		for k in d.keys():
			if type( d[k] )==np.ndarray:
				d[k] = dict.fromkeys( masks )
			else:
				clean(d[k])

	results_dict = deepcopy(explanations['attributions'])
	clean(results_dict)
	return results_dict


class Trainer():

	def __init__(self, model, lr=0.001,
	             criterion=torch.nn.CrossEntropyLoss(reduction='none')):

		self.model = model
		self.criterion = criterion
		# TODO do it better
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

	def forward_epoch(self, data_loader, training=False):

		mean_loss = 0.0
		mean_accuracy = 0.0
		tot_out = []

		if not training:
			self.model.eval()

		for i, batch_data in enumerate(data_loader):

			# get data, run forward
			X, y = batch_data
			if training:
				self.optimizer.zero_grad()
			out = self.model(X)

			# collect results
			tot_out.append(out)
			# TODO accuracy only if working in classification scenario
			mean_accuracy += (torch.argmax(out, dim=-1) == y).sum()

			# compute loss and collect results
			loss = self.criterion(out, y)
			if training:
				loss.mean().backward()
				self.optimizer.step()
			mean_loss += loss.sum().item()

		# concat single batch outputs and update stats
		tot_out = torch.concat(tot_out)
		mean_loss /= tot_out.shape[0]
		mean_accuracy /= tot_out.shape[0]

		return tot_out, mean_loss, mean_accuracy

	def train(self, train_loader, test_loader, model_path, n_epochs=150, n_epochs_stop=50):

		def print_stats():
			print('Epoch {}: train loss: {: .3f}, \t train accuracy {: .3f} \n'
			      '          test loss: {: .3f},  \t test accuracy {: .3f}'.format(
				epoch + 1, train_loss, train_accuracy, best_test_loss, best_test_accuracy,
			))

		# variable to be used
		best_test_accuracy = 0.0
		best_test_loss = np.inf
		non_improving_epochs = 0

		for epoch in range(n_epochs):

			# train and test
			out, train_loss, train_accuracy = self.forward_epoch(train_loader, training=True)
			test_out, test_loss, current_test_accuracy = self.test(test_loader)

			# early stopping
			if current_test_accuracy > best_test_accuracy:
				# TODO do i need all these test_out as in .test or in .fowrard_epoch?
				torch.save(self.model, model_path)
				best_test_accuracy = current_test_accuracy
				best_test_loss = test_loss
				non_improving_epochs = 0
			else:
				non_improving_epochs += 1

			if non_improving_epochs == n_epochs_stop or best_test_accuracy==1.0:
				print("training early stopped! Final stats are:")
				print_stats()
				break

			# print stats every 10 epochs
			if epoch % 10 == 0:
				print_stats()

		return test_out, best_test_accuracy

	def test(self, test_loader):

		with torch.no_grad():
			test_out, test_loss, test_accuracy = self.forward_epoch(test_loader, training=False)

		return test_out, test_loss, test_accuracy

def change_points_to_lengths(change_points, array_length):
    # change points is 1D iterable of idxs
    # assumes that each change point is the start of a new segment, aka change_points = start points
    start_points = np.array(change_points)
    end_points = np.append(change_points[1:], [array_length])
    #print(start_points, end_points)
    lengths = end_points - start_points
    return lengths

def lengths_to_weights(lengths):
    # lengths is 1D iterable of positive ints
    start_idx = 0
    end_idx = 0
    segment_weights = 1 / lengths
    weights = np.ones(lengths.sum())
    for segment_weight, length in zip(segment_weights, lengths):
        end_idx += length
        weights[start_idx: end_idx] = segment_weight
        start_idx = end_idx
    return weights