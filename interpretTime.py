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
    # TODO hardcoded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dataset name in the attributions file!
    explanations = np.load("attributions/all_results_resNet.npy", allow_pickle=True).item()
    #explanations = np.load( os.path.join("attributions" ,file_name+".npy") ,allow_pickle=True).item()
    # add a random explanation

    #TODO hardcoede!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ks = list(explanations[dataset_name].keys())
    ks.remove('y_test_true') ; ks.remove('label_mapping')  ; ks.append('random')     #explanations[dataset_name][ks[1]][classifier_name]['attributions']
    # add random exp
    rand = np.random.normal(loc=0,scale=1,size=X_test.shape)
    explanations[dataset_name]['random'] = {
        classifier_name:{
            'attributions' :
                {
                    'zero':     {'default': rand, 'normalized':rand},
                    'average':  {'default': rand, 'normalized':rand},
                    'sampling':  {'default': rand, 'normalized':rand},
                }
        }
    }

    # for each explanation and for each mask
    #TODO find the more convenient oreder
    import sys
    for nt in  ["normal_distribution","zeros","global_mean","local_mean","global_gaussian","local_gaussian"]:
        for k in ks:
            print("assessing ", k, "using",nt)

            # load model and explanations to access
            model_path = os.path.join("trained_models", file_name)
            for background in ['zero', 'average', 'sampling']:
                for result_type in ['default', 'normalized']:

                    explanations[dataset_name][k][classifier_name]['attributions'][background][result_type]

                    # TODO consider regression case i.e. no label!
                    manipulation_results = ScoreComputation(model_path=model_path,  clf_name = classifier_name,
                            background = background , result_type = result_type ,noise_type=nt,
                            encoder=explanations[dataset_name]['label_mapping'], data_dict = test_set_dict, device=device )

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
