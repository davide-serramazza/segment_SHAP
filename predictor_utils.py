import numpy as np
import torch
from pickle import load as load_sklearn
from torch import load as load_torch
from torch.nn import Sequential, Softmax
from sklearn.base import BaseEstimator as sklearn_Estimator
from torch.nn import Module as torch_Estimator

def load_predictor(path:str ,device='cpu'):
	"""
	Load predictor model base on path but WE CAN CHANGE IT VERY EASILY!

	:param path:
	:param device: device to be used for computation.It is only used for torch models,
	specifically "CPU" (CPU mode) or "cuda" (the model wil run on GPU)

	:return: the pre-trained predictor
	"""
	if path.endswith('.pkl'):
		# in this case this is a sklearn predictor (currently random Forest, QUANT and miniRocket)
		with open(path, 'rb') as f:
			predictor = load_sklearn(f)
	elif path.endswith('.pt'):
		# in this case this is a torch predictor (currently resNet)
		predictor = load_torch(path, map_location=device)
		predictor = Sequential(predictor, Softmax(dim=-1)).eval()
	else:
		raise ValueError(f'Unsupported predictor extension: {path}')

	return predictor


def predict_proba (clf, samples: np.ndarray, device='cpu') -> np.ndarray:
	"""
	Predict class probabilities from a pre-trained classifier.

	:param clf: classifier
	:param samples: samples to be predicted

	:return: probabilities as numpy arrays
	"""

	if isinstance(clf, sklearn_Estimator):
		# in this case simply call predict_proba method
		probas = clf.predict_proba(samples)
	elif isinstance(clf,torch_Estimator):
		# for torch first of all convert to torch Tensor
		samples = torch.tensor(samples).to(device)
		# then execute the forward in batch (currently set to 32)
		batch_size = 32
		probas = []
		for i in range(0, samples.shape[0], batch_size):
			current_samples = samples[i: min(i+batch_size,samples.shape[0] )]
			# append the proba results into a list AFTER converting them to numpy
			probas.append( clf(current_samples).detach().cpu().numpy() )
		# then concatenate individual forwards
		probas = np.concatenate(probas)
	else:
		raise ValueError(f'Unsupported predictor type: {clf}')

	return probas