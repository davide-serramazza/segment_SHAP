import numpy as np
import torch
from pickle import load as load_sklearn
from torch import load as load_torch
from torch.nn import Sequential, Softmax
from sklearn.base import BaseEstimator as sklearn_Estimator
from torch.nn import Module as torch_Estimator
import os
from sklearn.ensemble import RandomForestClassifier

def load_predictor(path:str, predictor_name:str, dataset_name:str ,device='cpu'):
	"""
	Load predictor model

	:param path:p   path to trained model folder
	:param device:  device to be used for computation.It is only used for torch models,
	specifically "CPU" (CPU mode) or "cuda" (the model wil run on GPU)

	:return: the pre-trained predictor
	"""
	model_path = os.path.join(path, "_".join( (predictor_name,dataset_name) ) )
	if os.path.isfile(model_path+".pkl"):
		# in this case this is a sklearn predictor (currently random Forest, QUANT and miniRocket)
		with open(model_path+".pkl", 'rb') as f:
			predictor = load_sklearn(f)

		# TODO TEMP WORKAROUND!
		if predictor_name=="randomForest":
			predictor = RandomForest(predictor)

	elif os.path.isfile(model_path+".pt"):
		# in this case this is a torch predictor (currently resNet)
		predictor = load_torch(model_path+".pt", map_location=device)
		predictor = Sequential(predictor, Softmax(dim=-1)).eval()
	else:
		raise ValueError(f'Unsupported predictor extension: {path}')

	return predictor


def predict_proba (clf, samples: np.ndarray, device='cpu') -> np.ndarray:
	"""
	Predict class probabilities from a pre-trained classifier.

	:param clf:     classifier
	:param samples: samples to be predicted

	:return:        probabilities as numpy arrays
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




# TEMP CLASS!

class RandomForest(RandomForestClassifier):

	def __init__(self,model):
		self.model = model

	def fit(self, X, y):
		X = np.reshape(X, ( X.shape[0] ,-1))
		self.model.fit(X, y)

	def predict_proba(self, X):
		X = np.reshape(X, ( X.shape[0] ,-1))
		return self.model.predict_proba(X)

