import os
import numpy as np
import torch.cuda
import timeit
import argparse
from load_data import  load_data
from utils import extract_InterpretTime_info
from InterpretTime.src.postprocessing_pytorch.manipulation_results import ScoreComputation
from InterpretTime.src.shared_utils.utils_visualization import plot_DeltaS_results, plot_additional_results


all_qfeatures = [0.05, 0.15, 0.25, 0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.0]

def main(args):

    dataset_name = args.datasets
    classifier_name = args.classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train, X_test, y_train, y_test, enc = load_data(subset='all', dataset_name=dataset_name)
    test_set_dict = extract_InterpretTime_info(X_test, X_train, dataset_name, y_test, y_train)

    # load explanations
    file_name = "_".join ( (classifier_name,dataset_name) )
    explanations = np.load( os.path.join("attributions" ,file_name+".npy") ,allow_pickle=True).item()
    # add a random explanation
    explanations['attributions']['rand'] = np.random.normal(loc=0,scale=1,size=X_test.shape)

    # for each explanation and for each mask
    for k in explanations['attributions'].keys():
        for nt in  ["normal_distribution","zeros","global_mean","local_mean","global_gaussian","local_gaussian"]:
            print("assessing ", k)

            # load model and explanations to access
            model_path = os.path.join("trained_models", file_name)
            attributions = explanations['attributions'][k]

            # TODO consider regression case i.e. no label!
            manipulation_results = ScoreComputation(model_path=model_path, noise_type=nt, clf_name = classifier_name,
                    encoder=explanations['label_mapping'], data_dict = test_set_dict, device=device )

            # TODO check whether to run out of or in the loop
            _ = manipulation_results.compute_scores_wrapper( all_qfeatures, k, attributions)
            manipulation_results.create_summary(k)

            manipulation_results.summarise_results()

            # save results and plot additional info
            #save_results_path = manipulation_results.save_results
            #plot_DeltaS_results(save_results_path)
            #plot_additional_results(save_results_path)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str, help="which dataset to be used")
    parser.add_argument("classifier", type=str, help="which predictor to be explained")
    args = parser.parse_args()
    main(args)
