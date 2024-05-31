from load_data import load_data
import matplotlib.pyplot as plt
import random
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import pandas as pd
import  numpy as np


for dataset_name in  ['gunpoint','UWAVE']:#, ] :
	# load metadata and data
	metadata = np.load("attributions/"+dataset_name+".npy", allow_pickle=True).item()
	segmentation = metadata['segments']
	X_test, y_test = load_data(subset='test', dataset_name=dataset_name)

	dir = "/".join( ("plots",dataset_name,"" ) )
	for i,segments in enumerate(segmentation):
		problems = [np.any(s)==[0] for s in segments]

		# plot each segment
		for j in range(X_test.shape[1]):
			ts = X_test[i,j]
			y=pd.Series(ts)
			change_points= segments[j]
			seg_name = "segmentation_"+str(i)+"_"+str(j)+".png"
			seg = plot_series_with_change_points(title= seg_name , y=y, change_points=change_points)
			seg[0].savefig(dir+seg_name)
			seg[0].clf()

		# if there is an empty segmentation plot all channels in the sample
		if np.any(problems):
			for j in range(len(problems)):
				ts = X_test[i,j]
				label = str(j)+"_PROBLEM" if problems[j]==True else str(j)
				plt.plot(ts,label=label)
			plt.legend()
			name = str(i)+".png"
			plt.savefig(dir+name)
			plt.clf()

