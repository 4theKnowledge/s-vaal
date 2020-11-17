"""
Main script which orchestrates active learning.

TODO
    - Migrate code into experimenter
    - Add incremental writing to JSON rather than writing at end of AL routine

@author: Tyler Bikaun
"""

import yaml
import random
import numpy as np
import math
import json

import torch
import torch.utils.data as data

from train import Trainer
from sampler import Sampler

class ActiveLearner(Trainer, Sampler):

    def __init__(self, config):
        Trainer.__init__(self, config)
        self.initial_budget_frac = 0.10 # fraction of samples that AL starts with
        self.val_frac = 0.05
        self.test_frac = 0.05
        self.budget_frac = 0.10
        self.data_splits_frac = np.round(np.linspace(self.budget_frac, 1, num=10, endpoint=True), 1)
        self.batch_size=64

    def _init_al_dataset(self):
        """ Initialises dataset for active learning
        """

        self._init_dataset()

        train_dataset = self.datasets['train']

        dataset_size = len(train_dataset)
        self.budget = math.ceil(self.budget_frac*dataset_size)
        Sampler.__init__(self, config, self.budget)     # TODO: Weird place to initialise this

        all_indices = set(np.arange(dataset_size))
        k_initial = math.ceil(len(all_indices)*self.initial_budget_frac)
        initial_indices = random.sample(list(all_indices), k=k_initial)

        sampler_init = data.sampler.SubsetRandomSampler(initial_indices)    # need to sample from training dataset

        self.labelled_dataloader = data.DataLoader(train_dataset, sampler=sampler_init, batch_size=self.batch_size, drop_last=True)
        self.val_dataloader = data.DataLoader(self.datasets['valid'], batch_size=self.batch_size, drop_last=False)
        self.test_dataloader = data.DataLoader(self.datasets['test'], batch_size=self.batch_size, drop_last=False)

        return all_indices, initial_indices

    def learn(self):
        """ Performs the active learning cycle """
        metrics_hist = dict()
        max_runs = 3
        for run in range(max_runs):
            all_indices, initial_indices = self._init_al_dataset()

            metrics_hist[str(run)] = dict()

            current_indices = list(initial_indices)
            
            for split in self.data_splits_frac:
                print(f'\nRUN {run} - SPLIT - {split*100:0.0f}%')

                # Initialise models
                self._init_models(mode='svaal')

                # Do some label stuff
                unlabelled_indices = np.setdiff1d(list(all_indices), current_indices)
                unlabelled_sampler = data.sampler.SubsetRandomSampler(unlabelled_indices)
                unlabelled_dataloader = data.DataLoader(self.datasets['train'],
                                                        sampler=unlabelled_sampler,
                                                        batch_size=64,
                                                        drop_last=False)

                print(f'Labelled: {len(current_indices)} Unlabelled: {len(unlabelled_indices)} Total: {len(all_indices)}')

                # TODO: Make the SVAAL allow 100% labelled and 0% unlabelled to pass through it. Breaking out of loop for now when data hits 100% labelled.
                if len(unlabelled_indices) == 0:
                    break

                metrics, svae, discriminator = self.train(dataloader_l=self.labelled_dataloader,
                                                            dataloader_u=unlabelled_dataloader,
                                                            dataloader_v=self.val_dataloader,
                                                            dataloader_t=self.test_dataloader,
                                                            mode='svaal')                                                        
                print(f'Test Eval.: F1 Scores - Macro {metrics[0]*100:0.2f}% Micro {metrics[1]*100:0.2f}%')        
                
                # Record performance at each split
                metrics_hist[str(run)][str(split)] = metrics

                
                sampled_indices = self.sample_adversarial(svae, discriminator, unlabelled_dataloader, indices=unlabelled_indices, cuda=True)    # TODO: review usage of indices arg
                current_indices = list(current_indices) + list(sampled_indices)
                sampler = data.sampler.SubsetRandomSampler(current_indices)
                self.labelled_dataloader = data.DataLoader(self.datasets['train'], sampler=sampler, batch_size=self.batch_size, drop_last=True)
            
        # write results to disk
        with open('results.json', 'w') as fj:
            json.dump(metrics_hist, fj, indent=4)


def main(config):
    
    al = ActiveLearner(config)
    al.learn()

if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    # Seeds
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed'])

    main(config)