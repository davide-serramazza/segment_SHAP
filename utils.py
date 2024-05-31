import numpy as np
import torch

def forward_classification(X_test : torch.Tensor, model):
	# convert X to pytorch tensor
	X_test_numpy = X_test.detach().numpy()
	# compute probability
	predictions = model.predict_proba(X_test_numpy)
	# return result as torch tensor as expected by captum attribution method
	return torch.tensor(predictions)

def sample_background(X_train,n):
	to_select = np.random.permutation(X_train.shape[0])[:n]
	return torch.tensor( X_train[to_select] )