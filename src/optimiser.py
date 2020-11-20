"""
Runs optimisation of hyperparameters of neural networks. This will be fleshed out to use Bayesian optimisation.

@author: Tyler Bikaun
"""

import numpy as np
import yaml
from ax import optimize


import torch


from experimenter import Experimenter

def _opt_full_data_performance(config):
    """"""

    exp = Experimenter(config)

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
        )

    return best_parameters




if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    # Seeds
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed'])

    best_params = _opt_full_data_performance(config)
    print(best_params)