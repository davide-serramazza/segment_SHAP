import os
import pickle

import numpy as np
import torch.cuda
import timeit
import pandas as pd
import argparse
import itertools
from load_data import  load_data
from utils import extract_InterpretTime_info
from InterpretTime.src.postprocessing_pytorch.manipulation_results import ScoreComputation
from InterpretTime.src.shared_utils.utils_visualization import plot_DeltaS_results, plot_additional_results
from pickle import dump
from utils import intantiate_dict_results

def main(args):

    # passing args and device
    dataset_name = args.datasets
    classifier_name = args.classifier
    demo_mode = args.demo_mode
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading dataset
    all_qfeatures = [0.05 ,1.0] if args.demo_mode else [0.05, 0.15, 0.25, 0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.0]
    X_train, X_test, y_train, y_test, enc = load_data(subset='all', dataset_name=dataset_name)
    if demo_mode:
        X_test, y_test = X_test[:2], y_test[:2]
    test_set_dict = extract_InterpretTime_info(X_test, X_train, dataset_name, y_test, y_train)

    # load explanations
    file_name = "_".join ( (classifier_name,dataset_name) )
    attribution_name = "_".join( ("all_results",dataset_name,classifier_name) ) + ".npy"
    explanations = np.load(os.path.join("attributions",attribution_name), allow_pickle=True).item()

    # get infos about which explanations are evaluated
    datasets = list( explanations['attributions'].keys() )
    segmentations = list( explanations['attributions'][datasets[0]].keys() )
    predictors = list( explanations['attributions'][datasets[0]][segmentations[0]] .keys() )
    backgrounds = list( explanations['attributions'][datasets[0]][segmentations[0]][predictors[0]].keys() )
    result_types = ['default','normalized']
    masks = ["normal_distribution","zeros","global_mean","local_mean","global_gaussian","local_gaussian"]

    # initialize data structure to store results
    results_dict = intantiate_dict_results(explanations,masks)

    starttime = timeit.default_timer()
    for it in itertools.product(datasets,segmentations,predictors,backgrounds,result_types,masks):
        dataset,segmentation,predictor,background,result_type,mask = it
        print("assessing ", it[:-1], "using",mask)

        # load model and explanations to access
        model_path = os.path.join("trained_models", file_name)
        attributions = explanations['attributions'][dataset][segmentation][predictor][background][result_type]

        # TODO consider regression case i.e. no label!
        # run Interpret time
        isRF = classifier_name=="randomForest"
        result_path = os.path.join(dataset,segmentation,predictor,background,result_type,mask)
        manipulation_results = ScoreComputation(model_path=model_path, result_path=result_path ,noise_type=mask,
            randomForest=isRF, encoder=explanations['label_mapping'][dataset], data_dict = test_set_dict, device=device)

        manipulation_results.compute_scores_wrapper( all_qfeatures, segmentation, attributions)
        manipulation_results.create_summary(segmentation)

        # store results in the nested data structure
        current_result = manipulation_results.summarise_results()
        results_dict[dataset][segmentation][predictor][background][result_type][mask] = {
            'AUCSE_top' : current_result['AUCSE_top'][segmentation],
            'F_score' : current_result['F_score'][segmentation]
        }
        # save results and plot additional info
        #save_results_path = manipulation_results.save_results
        #plot_DeltaS_results(save_results_path)
        #plot_additional_results(save_results_path)

    res_file_name = "demo_dict_result" if demo_mode else "dict_result"
    with open( "_".join( (res_file_name,dataset_name,classifier_name)) ,"wb") as f:
        pickle.dump(results_dict,f)

    print("elapsed time", ( timeit.default_timer() -starttime ) )


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str, help="which dataset to be used")
    parser.add_argument("classifier", type=str, help="which predictor to be explained")
    parser.add_argument("demo_mode", type=str, nargs='?', help="whether demo mode i.e. only two samples")
    args = parser.parse_args()
    args.demo_mode = True if args.demo_mode is not None and args.demo_mode.lower() =="true" else False
    main(args)
