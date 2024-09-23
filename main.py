from captum.attr import ShapleyValueSampling
from tqdm import trange

from load_data import load_data
from models.train_models import train_randomForest
from segmentation import *
from utils import *

def main():

	# load data
	dataset_name = 'UWAVE'
	X_train, X_test, y_train, y_test = load_data(subset='all', dataset_name=dataset_name)

	# train model
	clf, preds = train_randomForest(X_train,y_train,X_test,y_test, dataset_name)

	# create a dictionary to be dumped containing attribution and metadata
	# initialize data structure meant to contain the segments
	segments =  np.empty( (X_test.shape[0] , X_test.shape[1]), dtype=object) if X_test.shape[1] > 1  else (
		np.empty( X_test.shape[0] , dtype=object))

	all_attributions = {
		'attributions' : np.empty( X_test.shape ,dtype=np.float32 ),
		'segments' : segments,
		'y_test_true' : y_test,
		'y_test_pred' : preds
	}

	# explain
	batch_size = 32
	with torch.no_grad():
		SHAP = ShapleyValueSampling(forward_classification)
		for i in range ( X_test.shape[0] ) :
			# get current sample and label
			ts, y = X_test[i] , torch.tensor( y_test[i:i+1] )

			# get segment and its tensor representation
			current_segments = get_claSP_segmentation(ts)[:X_test.shape[1]]
			all_attributions['segments'][i] = current_segments
			mask = get_feature_mask(current_segments,ts.shape[-1])

			# background data
			background_dataset = sample_background(X_train,50)

			print("\n explaining sample n.",i,"\n")
			# data structure with room for each sample in the background dataset
			current_attr = torch.zeros(background_dataset.shape[0], ts.shape[0], ts.shape[1])
			for j in trange(0,background_dataset.shape[0] ,batch_size):

				sample = background_dataset[j:j+batch_size]
				actual_size = sample.shape[0]
				batched_ts = torch.tensor( [ts]*actual_size )

				##### only for random forest as every instance should be a 1D tensor #######
				batched_ts , sample = batched_ts.reshape(actual_size,-1), sample.reshape(actual_size,-1)
				mask = mask.reshape(1,-1)
				###############################################################################

				tmp = SHAP.attribute( batched_ts, target=y , feature_mask=mask, baselines=sample, additional_forward_args=clf)

				########  only for random forest as every instance should be a 1D tensor    ########
				current_attr[j:j+actual_size] = tmp.reshape(actual_size,X_test.shape[1],X_test.shape[2])
				###############################################################################

			# compute as final explanation mean of each explanation using a different baseline
			all_attributions['attributions'][i] =torch.mean(current_attr,dim=0)

	# dump result to disk
	np.save("attributions/"+dataset_name, all_attributions)

if __name__ == '__main__':
	main()