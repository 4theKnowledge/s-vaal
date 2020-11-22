"""
Runs optimisation of hyperparameters of neural networks. This will be fleshed out to use Bayesian optimisation.

@author: Tyler Bikaun
"""

import numpy as np
import yaml
from ax import optimize

import torch

from experimenter import Experimenter
from connections import load_config

def _opt_full_data_performance():
    """"""

    exp = Experimenter()

    best_parameters, best_values, experiment, model = optimize(
            parameters=[
            {
                "name": "embedding_dim",
                "type": "range",
                "bounds": [128, 1024],
            },
            {
                "name": "hidden_dim",
                "type": "range",
                "bounds": [128, 1024],
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [16, 64],
            },
            ],
            evaluation_function= exp._full_data_performance,
            minimize=False,
            objective_name='f1_macro',
            total_trials=3
        )

    return best_parameters

def _opt_svae():

    exp = Experimenter()

    best_parameters, best_values, experiment, model = optimize(
            parameters=[
            {
                "name": "embedding_dim",
                "type": "range",
                "bounds": [128, 1024],
            },
            {
                "name": "hidden_dim",
                "type": "range",
                "bounds": [128, 1024],
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [16, 64],
            },
            {
                "name": "rnn_Type",
                "type": "string",
                "bounds": ["lstm", "gru"],
            },
            {
                "name": "num_layers",
                "type": "range",
                "bounds": [1, 1],
            },
            {
                "name": "bidirectional",
                "type": "bool",
                "bounds": [True],
            },
            {
                "name": "latent_size",
                "type": "range",
                "bounds": [16, 256],
            },
            {
                "name": "word_dropout",
                "type": "range",
                "bounds": [0, 1],
            },
            {
                "name": "embedding_dropout",
                "type": "range",
                "bounds": [0, 1],
            },
            ],
            evaluation_function= exp._svae,
            minimize=False,
            objective_name='val_loss',
        )

    return best_parameters



if __name__ == '__main__':
    # Seeds
    config = load_config()
    np.random.seed(config['Train']['seed'])
    torch.manual_seed(config['Train']['seed'])

    best_params = _opt_full_data_performance()
    print(best_params)