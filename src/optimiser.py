"""
Runs optimisation of hyperparameters of neural networks. This will be fleshed out to use Bayesian optimisation.

TODO:
    - Add learning rate to optimiser? Currently models have learning rate decay schedulers, but optim could find best
      starting position.
    - Add number of epochs

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
    # TODO: Add anneal function, x0 and k to optimiser + epochs

    exp = Experimenter()

    best_parameters, best_values, experiment, model = optimize(
            parameters=[
            {
                "name": "epochs",
                "type": "range",
                "value_type": "int",
                "bounds": [25,100],
            },
            {
                "name": "embedding_dim",
                "type": "range",
                "value_type": "int",
                "bounds": [128, 1024],
            },
            {
                "name": "hidden_dim",
                "type": "range",
                "value_type": "int",
                "bounds": [128, 1024],
            },
            {
                "name": "batch_size",
                "type": "range",
                "value_type": "int",
                "bounds": [16, 64],
            },
            {
                "name": "rnn_type",
                "type": "fixed",
                "value_type": "str",
                "value": "gru",
            },
            {
                "name": "num_layers",
                "type": "fixed",
                "value_type": "int",
                "value": 1,
            },
            {
                "name": "bidirectional",
                "type": "fixed",
                "value_type": "bool",
                "value": True,
            },
            {
                "name": "latent_size",
                "type": "range",
                "value_type": "int",
                "bounds": [16, 256],
            },
            {
                "name": "word_dropout",
                "type": "range",
                "value_type": "float",
                "bounds": [0, 1],
            },
            {
                "name": "embedding_dropout",
                "type": "range",
                "value_type": "float",
                "bounds": [0, 1],
            },
            ],
            evaluation_function= exp._svae,
            minimize=True,
            objective_name='train_loss',
            total_trials=10
        )

    return best_parameters



if __name__ == '__main__':
    # Seeds
    # config = load_config()
    # np.random.seed(config['Train']['seed'])
    # torch.manual_seed(config['Train']['seed'])



    # best_params = _opt_full_data_performance()
    # print(best_params)

    best_params = _opt_svae()
    print(best_params)