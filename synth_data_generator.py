#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Originally Written by H.Turbé, June 2022.

"""
import os
import sys
import warnings

import numpy as np
warnings.simplefilter(action="ignore", category=FutureWarning)
FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILEPATH)

#from shared_types.schema import ShapeSchema


class PreprocessSynthetic:
    """
    Main class to generate the synthetic dataset
    """
    def __init__(self, data_config):
        self.data_config = data_config

        self.n_points = data_config["properties"]["n_points"]
        self.n_support = data_config["properties"]["n_support"]
        self.n_features = data_config["properties"]["n_features"]
        self.f_min_support, self.f_max_support = data_config["properties"]["f_sin"]
        self.f_min_base, self.f_max_base = data_config["properties"]["f_base"]
        #self.lines = data_config["properties"]["line_types"]
        # we create a distribution for the sum of sine frequency to
        # have control over class distribution
        f_sine_1 = np.random.randint(
            self.f_min_support, (self.f_max_support + 1), 10000
        )
        f_base_1 = np.random.randint(self.f_min_base, (self.f_max_base + 1), 10000)
        f_sine_2 = np.random.randint(
            self.f_min_support, (self.f_max_support + 1), 10000
        )
        f_base_2 = np.random.randint(self.f_min_base, (self.f_max_base + 1), 10000)
        f_sum = f_sine_1 + f_sine_2
        self.quantile_class_sum = np.quantile(
            f_sum, data_config["properties"]["quantile_class"]
        )
        f_ratio = (f_sine_1 / f_base_1) + (f_sine_2 / f_base_2)
        self.quantile_class_ratio = np.quantile(
            f_ratio, data_config["properties"]["quantile_class"]
        )


    def _generate_feature(self, wave: str, f_support: int, f_base: float):
        """
        Generate a given feature of a sample with a support of a given type and frequency
        overlaid over a sine wave of a given frequency
        Parameters
        ----------
        wave: str
            type of the support wave
        f_support: int
            frequency of the support wave
        f_base: float
            frequency of the base wave
        Returns
        -------
        x_feature: np.array
            feature of the sample
        start_idx: int
            idx where the support starts
        """
        x_feature = np.sin(np.linspace(0, 2 * np.pi * f_base, self.n_points)).reshape(
            -1, 1
        )
        x_feature *= 0.5
        start_idx = np.random.randint(0, self.n_points - self.n_support)
        wave_length = 0

        if wave == "sine":
            x_tmp = np.sin(
                np.linspace(0, 2 * np.pi * f_support, self.n_points)
            ).reshape(-1, 1)
            start_tmp = 0
            wave_length = np.ceil(self.n_points / f_support).astype(int)
            x_feature[start_idx : start_idx + wave_length, 0] += x_tmp[
                                                                    start_tmp : start_tmp + wave_length, 0
                                                                    ]

            #x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
            #                                                        start_tmp : start_tmp + self.n_support, 0
            #                                                        ]

        elif wave == "square":
            x_tmp = np.sign(
                np.sin(np.linspace(0, 2 * np.pi * f_support, self.n_points))
            ).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                                                                    start_tmp : start_tmp + self.n_support, 0
                                                                    ]

        elif wave == "line":
            x_feature[start_idx : start_idx + self.n_support, 0] += [0] * self.n_support
        else:
            raise ValueError("wave must be one of sine, square, sawtooth, line")
        return x_feature, start_idx, wave_length

    def _generate_sample(self, index):
        """
        Generate a sample
        Parameters
        ----------
        index: int
            index of the sample
        Returns
        -------
        dict_all: dict
            dictionary containing the sample with the information saved
            in parquet format
        """
        dict_all = {}
        dict_all["noun_id"] = f"sample_{index}"
        idx_features = np.random.permutation(np.arange(self.n_features))
        x_sample = np.zeros((self.n_points, self.n_features))
        f_sine_sum = 0
        f_ratio = 0
        pos_sin_wave = []
        wave_lengths = []
        for enum, idx_feature in enumerate(idx_features):
            f_base = np.random.randint(self.f_min_base, (self.f_max_base + 1), 1)[0]
            f_support = np.random.randint(
                self.f_min_support, (self.f_max_support + 1), 1
            )[0]
            if enum < 2:

                f_sine_sum += f_support
                f_ratio += f_support / f_base
                x_tmp, start_idx, wave_length = self._generate_feature(
                    wave="sine", f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()
                pos_sin_wave.append(start_idx)
                wave_lengths.append(wave_length)
            else:

                #wave = random.choice(self.lines)
                x_tmp, _ , _ = self._generate_feature(  #wave=wave,
                    wave='line', f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()

        dict_all["signal"] = x_sample.astype(np.float32)

        dict_all["target_sum"] = (
            np.argwhere(f_sine_sum <= self.quantile_class_sum).min().astype(str)
        )

        bool_class_ratio = f_ratio <= self.quantile_class_ratio
        if np.sum(bool_class_ratio) == 0:
            dict_all["target_ratio"] = str(len(self.quantile_class_ratio) - 1)
        else:
            dict_all["target_ratio"] = np.argwhere(bool_class_ratio).min().astype(str)

        dict_all["signal_length"] = x_sample.shape[0]
        dict_all["signal_names"] = np.array(
            [f"feature_{str(x)}" for x in range(x_sample.shape[1])]
        ).astype(np.string_)
        dict_all["pos_sin_wave"] = np.array(pos_sin_wave).astype(int)
        dict_all["feature_idx_sin_wave"] = idx_features[:2].astype(int)
        dict_all["wave_lengths"] = np.array(wave_lengths).astype(int)
        dict_all["f_sine_sum"] = f_sine_sum.astype(int)
        dict_all["f_sine_ratio"] = f_ratio.astype(np.float32)

        return dict_all


def get_ground_truth(instance_dict):
    """
    function to generate ground truth as a matrix containing either 0 or 1
    NOT USED AT THE MOMENT
    :param instance_dict:   dictionary containing all info about a synthetic time series
    :return:
    """
    gt = np.zeros(shape= instance_dict["signal"].transpose().shape)
    for j, channel in enumerate(instance_dict['feature_idx_sin_wave']):
        support_start = instance_dict['pos_sin_wave'][j]
        gt[ channel, support_start: (support_start + 100)] = 1
    return gt

def get_cps (wave_locations,channels, wave_lengths ):
    cps = np.array( [0]*6, dtype=object )     # TODO number of total channel is hardcoded
    for i in range(2): # n discriminative channels # TODO hardcoded
        current_channel = channels[i]
        current_cps = np.array( [ 0, wave_locations[i], wave_locations[i]+ wave_lengths[i] ] )  #TODO support wave length is hard coded
        cps[current_channel] = current_cps
    return cps

def generate_set(n_instances):
    """
    function that uses all other functions defined here to create a set (training, val or test)

    :param n_instances: number of instance for the current set
    :return: a dict structured in such a way {
        X :         features for this set
        'y_clf' :   label for classification i.e. whether the sum of thw two frequencies are above or below
                                                a specific threshold
        'y_reg' :   label for classification i.e. sum of the two wave's frequencies
        'cps' :     changing point i.e. when a new distribution (new wave) starts
        }
    """
    X = []
    y_clf = []
    y_reg = []
    cps = []
    #   y = []              NOT used at the moment
    #   ground_truths = []  NOT used at the moment
    additional_info = []
    for i in range(n_instances):
        instance_dict = pp._generate_sample(i)
        X.append(instance_dict["signal"].transpose())
        cps.append( get_cps(instance_dict['pos_sin_wave'],instance_dict['feature_idx_sin_wave'], instance_dict['wave_lengths'] ) )
        y_clf.append( instance_dict['target_sum'] )
        y_reg.append( instance_dict['f_sine_sum'] )
        #ground_truths.append(get_ground_truth(instance_dict))

        del instance_dict["signal"]
        additional_info.append(instance_dict)

    dataset = {'X': np.stack(X),
                'y_clf' : np.stack(y_clf),
                'y_reg' : np.stack(y_reg),
                'cps' : np.stack(cps)
             }
    return dataset

if __name__ == "__main__" :
    n_instances = [5000,1000]               # number of instances for respectively training and test set
    data_config = {
        'properties': {
            'n_points': 500,                # length of each TS channel
            'n_support': 100,               # length of the 'support wave' namely the discriminative part in TS
            'n_features': 6,                # number of channel (in this work feature is used as channel)
            'f_base': [2,5],                # range of frequency for the 'background waves' i.e. NON discriminative
            'f_sin': [10, 30],              # range of frequencies for the 'support waves' i.e. discriminative part
            'quantile_class': [0.5, 1],     # used for the classification threshold
        }
    }
    pp = PreprocessSynthetic(data_config=data_config)
    file_name = "synth_data.npy"
    path= "./datasets/"
    train = generate_set(n_instances[0])
    test = generate_set(n_instances[1])
    np.save( os.path.join(path,file_name), {
            'train' : train,
            'test' : test
        })
    print( "saved in ",  os.path.join(path,file_name) )