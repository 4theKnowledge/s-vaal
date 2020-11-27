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
from datetime import datetime 

import torch

from experimenter import Experimenter
from connections import load_config, Mongo

class Optimiser:
    def __init__(self):
        self.exp = Experimenter()
        self.mongo_coll_conn = Mongo(collection_name='optimisation')
        self.trials = 10
        self.config = load_config()
        self.task_type = self.config['Utils']['task_type']
        self.data_name = self.config['Utils'][self.task_type]['data_name']
        
    def _opt_full_data_performance(self, objective_name="f1_macro", minimise=None):
        """ Optimisation routine for full data performance of task learner """
        # TODO: Add Epochs
        start_time = datetime.now()
        best_parameters, best_values, _, _ = optimize(
                                                    parameters=[
                                                        {
                                                            "name": "epochs",
                                                            "type": "range",
                                                            "bounds": [16, 128]    
                                                        },
                                                        {
                                                            "name": "batch_size",
                                                            "type": "range",
                                                            "bounds": [16, 64],
                                                        },
                                                        {
                                                            "name": "tl_embedding_dim",
                                                            "type": "range",
                                                            "bounds": [128, 1024],
                                                        },
                                                        {
                                                            "name": "tl_hidden_dim",
                                                            "type": "range",
                                                            "bounds": [128, 1024],
                                                        },
                                                        {
                                                            "name": "learning_rate",    # TODO: Will need to update to tl_ in the future
                                                            "type": "range",
                                                            "bounds": [0.00001, 0.1]
                                                        },
                                                        ],
                                                        evaluation_function= self.exp._full_data_performance,
                                                        minimize=minimise,
                                                        objective_name=objective_name,
                                                        total_trials=self.trials
                                                )
        finish_time = datetime.now()
        run_time = (finish_time-start_time).total_seconds()/60

        # TODO: Will put into a decorated in the future...
        data = {"name": f"FDP-{self.task_type}-{self.data_name}",
                "info": {"start timestamp": start_time,
                         "finish timestamp": finish_time,
                         "run time": run_time},
                "config": self.config,
                "settings": {"trials": self.trials, "object name": objective_name, "minimise": minimise},
                "results": {"best parameters": best_parameters, "best value": best_values[0][objective_name]}}
        # # Post results to mongodb
        self.mongo_coll_conn.post(data)
        
    def _opt_svae(self, objective_name="train_loss", minimise=None):
        """ Optimisation routine for full data performance of SVAE """
        
        start_time = datetime.now()
        best_parameters, best_values, _, _ = optimize(
                                                    parameters=[
                                                    {
                                                        "name": "epochs",
                                                        "type": "range",
                                                        "value_type": "int",
                                                        "bounds": [25,100],
                                                    },
                                                    {
                                                        "name": "batch_size",
                                                        "type": "range",
                                                        "value_type": "int",
                                                        "bounds": [16, 64],
                                                    },
                                                    {
                                                        "name": "svae_embedding_dim",
                                                        "type": "range",
                                                        "value_type": "int",
                                                        "bounds": [128, 1024],
                                                    },
                                                    {
                                                        "name": "svae_hidden_dim",
                                                        "type": "range",
                                                        "value_type": "int",
                                                        "bounds": [128, 1024],
                                                    },
                                                    {
                                                        "name": "svae_rnn_type",
                                                        "type": "fixed",
                                                        "value_type": "str",
                                                        "value": "gru",
                                                    },
                                                    {
                                                        "name": "svae_num_layers",
                                                        "type": "fixed",
                                                        "value_type": "int",
                                                        "value": 1,
                                                    },
                                                    {
                                                        "name": "svae_bidirectional",
                                                        "type": "fixed",
                                                        "value_type": "bool",
                                                        "value": True,
                                                    },
                                                    {
                                                        "name": "svae_latent_size",
                                                        "type": "range",
                                                        "value_type": "int",
                                                        "bounds": [16, 256],
                                                    },
                                                    {
                                                        "name": "svae_word_dropout",
                                                        "type": "range",
                                                        "value_type": "float",
                                                        "bounds": [0, 1],
                                                    },
                                                    {
                                                        "name": "svae_embedding_dropout",
                                                        "type": "range",
                                                        "value_type": "float",
                                                        "bounds": [0, 1],
                                                    },
                                                    {
                                                        "name": "svae_k",
                                                        "type": "range",
                                                        "value_type": "float",
                                                        "bounds": [0, 1],
                                                    },
                                                    {
                                                        "name": "svae_x0",
                                                        "type": "range",
                                                        "value_type": "int",
                                                        "bounds": [250, 5000],
                                                    },
                                                    ],
                                                    evaluation_function= self.exp._svae,
                                                    minimize=minimise,
                                                    objective_name=objective_name,
                                                    total_trials=self.trials
                                                )
        finish_time = datetime.now()
        run_time = (finish_time-start_time).total_seconds()/60

        # TODO: Will put into a decorated in the future...
        data = {"name": "SVAE",
                "info": {"start timestamp": start_time,
                            "finish timestamp": finish_time,
                            "run time": run_time},
                "settings": {"trials": self.trials, "object name": objective_name, "minimise": minimise},
                "results": {"best parameters": best_parameters, "best value": best_values[0][objective_name]}}
        # # Post results to mongodb
        self.mongo_coll_conn.post(data)

    def _opt_svaal(self, objective_name="val_f1_macro", minimise=None):
        """ Optimisation routine for SVAAL model
        
        Notes
        -----
        This model differs from SVAE and FDP as SVAAL is not necessarily trained on full data.
        
        TODO: Include learning rates into optimisation
        """
        
        start_time = datetime.now()
        best_parameters, best_values, experiment, model = optimize(
                parameters=[
                {
                    "name": "epochs",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [25,100],
                },
                {
                    "name": "batch_size",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [16, 64],
                },
                {
                    "name": "tl_embedding_dim",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [128, 1024],
                },
                {
                    "name": "tl_hidden_dim",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [128, 1024],
                },
                {
                    "name": "tl_rnn_type",
                    "type": "fixed",
                    "value_type": "str",
                    "value": "gru",
                },
                {
                    "name": "disc_z_dim",
                    "type": "range",
                    "value_type": "int",
                    "value": [64,512],
                },
                {
                    "name": "svae_embedding_dim",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [128, 1024],
                },
                {
                    "name": "svae_hidden_dim",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [128, 1024],
                },
                {
                    "name": "svae_num_layers",
                    "type": "fixed",
                    "value_type": "int",
                    "value": 1,
                },
                {
                    "name": "svae_bidirectional",
                    "type": "fixed",
                    "value_type": "bool",
                    "value": True,
                },
                {
                    "name": "svae_latent_size",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [16, 256],
                },
                {
                    "name": "svae_word_dropout",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 1],
                },
                {
                    "name": "svae_embedding_dropout",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 1],
                },
                {
                    "name": "svae_k",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 1],
                },
                {
                    "name": "svae_x0",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [250, 5000],
                },
                {
                    "name": "svae_adv_hyperparameter",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0, 1],
                },
                ],
                evaluation_function= self.exp.learn,
                minimize=minimise,
                objective_name=objective_name,
                total_trials=self.trials
            )
        finish_time = datetime.now()
        run_time = (finish_time-start_time).total_seconds()/60

        # TODO: Will put into a decorated in the future...
        data = {"name": "SVAE",
                "info": {"start timestamp": start_time,
                            "finish timestamp": finish_time,
                            "run time": run_time},
                "settings": {"trials": self.trials, "object name": objective_name, "minimise": minimise},
                "results": {"best parameters": best_parameters, "best value": best_values[0][objective_name]}}
        # # Post results to mongodb
        self.mongo_coll_conn.post(data)

if __name__ == '__main__':
    # Seeds
    # config = load_config()
    # np.random.seed(config['Train']['seed'])
    # torch.manual_seed(config['Train']['seed'])

    # Optimisation routines
    Optimiser()._opt_full_data_performance(objective_name="f1_macro", minimise=False)

    # Optimiser()._opt_svae(objective_name="train_loss", minimise=True)

    # Optimiser()._opt_svaal(objective_name="val_f1_macro", minimise=False)