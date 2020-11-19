"""
Runs batch based experiments on models this will include running various active learning methods.

@author: Tyler Bikaun
"""

# TODO:
# - Add experiment naming, directory creation, parameter setting, etc.
# - Add tensorboard logging for active learning plots (metric vs. % data), etc.

import yaml
import numpy as np
import math
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from train import Trainer
from sampler import Sampler
from tasklearner import TaskLearner
from utils import trim_padded_seqs

class Experimenter(Trainer, Sampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        self.config = config

        self.initial_budget_frac = 0.10     # fraction of samples that AL starts with
        self.budget_frac = 0.10     # fraction of data to sample at each AL iteration
        self.data_splits_frac = np.round(np.linspace(self.budget_frac, 1, num=10, endpoint=True), 1)
        self.batch_size = config['Train']['batch_size']

        self.max_runs = 3

        self.al_mode = 'svaal'     # option: svaal, random

        self._setup_utils()

    def _setup_utils(self):
        """ Sets up utilities such as logging, saving, tensorboard and data recording/caching
        """
        self.tb_writer = SummaryWriter()

    def _init_al_data(self):
        """ Initialises train, validation and test sets for active learning including partitions

        Notes
        -----
        """
        # Initialse dataset
        self._init_dataset()

        train_dataset = self.datasets['train']

        dataset_size = len(train_dataset)
        self.budget = math.ceil(self.budget_frac*dataset_size)
        Sampler.__init__(self, self.config, self.budget)     # TODO: Weird place to initialise this, but whatever

        all_indices = set(np.arange(dataset_size))
        k_initial = math.ceil(len(all_indices)*self.initial_budget_frac)
        initial_indices = random.sample(list(all_indices), k=k_initial)

        sampler_init = data.sampler.SubsetRandomSampler(initial_indices)    # need to sample from training dataset

        self.labelled_dataloader = data.DataLoader(train_dataset, sampler=sampler_init, batch_size=self.batch_size, drop_last=True)
        self.val_dataloader = data.DataLoader(self.datasets['valid'], batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = data.DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=True, drop_last=False)

        print('----- DATA INITIALISED -----')

        return all_indices, initial_indices

    def learn(self):
        """ Performs active learning routine
        """

        metrics_hist = dict()
            
        for run in range(1, self.max_runs+1):
            all_indices, initial_indices = self._init_al_data()

            metrics_hist[str(run)] = dict()

            current_indices = list(initial_indices)
            
            for split in self.data_splits_frac:
                print(f'\nRUN {run} - SPLIT - {split*100:0.0f}%')

                # Initialise models
                if self.al_mode == 'svaal':
                    self._init_models(mode='svaal')
                elif self.al_mode == 'random':
                    self._init_models(mode=None)

                # Do some label stuff
                unlabelled_indices = np.setdiff1d(list(all_indices), current_indices)
                unlabelled_sampler = data.sampler.SubsetRandomSampler(unlabelled_indices)
                unlabelled_dataloader = data.DataLoader(self.datasets['train'],
                                                        sampler=unlabelled_sampler,
                                                        batch_size=32,
                                                        drop_last=False)

                print(f'Labelled: {len(current_indices)} Unlabelled: {len(unlabelled_indices)} Total: {len(all_indices)}')
                
                # TODO: Make the SVAAL allow 100% labelled and 0% unlabelled to pass through it. Breaking out of loop for now when data hits 100% labelled.
                if len(unlabelled_indices) == 0:
                    break

                # Perform AL training and sampling
                if self.al_mode == 'svaal':
                    metrics, sampled_indices = self._svaal(self.labelled_dataloader,
                                                            unlabelled_dataloader,
                                                            self.val_dataloader,
                                                            self.test_dataloader,
                                                            unlabelled_indices)
                elif self.al_mode == 'random':
                    metrics, sampled_indices = self._random_sampling(self.labelled_dataloader,
                                                                        unlabelled_dataloader,
                                                                        self.val_dataloader,
                                                                        self.test_dataloader,
                                                                        unlabelled_indices)

                print(f'Test Eval.: F1 Scores - Macro {metrics[0]*100:0.2f}% Micro {metrics[1]*100:0.2f}%')        
                
                # Record performance at each split
                metrics_hist[str(run)][str(split)] = metrics

                
                current_indices = list(current_indices) + list(sampled_indices)
                sampler = data.sampler.SubsetRandomSampler(current_indices)
                self.labelled_dataloader = data.DataLoader(self.datasets['train'], sampler=sampler, batch_size=self.batch_size, drop_last=True)

        # write results to disk
        with open('results.json', 'w') as fj:
            json.dump(metrics_hist, fj, indent=4)

    def _svaal(self, dataloader_l, dataloader_u, dataloader_v, dataloader_t, unlabelled_indices):
        """ S-VAAL routine

        Arguments
        ---------

        Returns
        -------

        Notes
        -----

        """
        metrics, svae, discriminator = self.train(dataloader_l=dataloader_l,
                                                    dataloader_u=dataloader_u,
                                                    dataloader_v=dataloader_v,
                                                    dataloader_t=dataloader_t,
                                                    mode='svaal')
        sampled_indices = self.sample_adversarial(svae,
                                                    discriminator,
                                                    dataloader_u,
                                                    indices=unlabelled_indices,
                                                    cuda=True)    # TODO: review usage of indices arg
        return metrics, sampled_indices

    def _full_data_performance(self):
        """ Gets performance of task learner on full dataset without any active learning
        """
        # Initialise data (need vocab etc)
        self._init_dataset()

        dataloader_l = data.DataLoader(dataset=self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Initialise model (TaskLearner only)
        task_learner = TaskLearner(**self.model_config['TaskLearner']['Parameters'], vocab_size=self.vocab_size, tagset_size=self.tag_space_size, task_type=self.task_type).to(self.device)
        if self.task_type == 'NER':
            tl_loss_fn = nn.NLLLoss().to(self.device)
        if self.task_type == 'CLF':
            tl_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Optimisers
        tl_optim = optim.SGD(task_learner.parameters(), lr=0.1, momentum=0, weight_decay=0.1)       # TODO: Update with params from config
        # Learning rate scheduler
        tl_sched = optim.lr_scheduler.ReduceLROnPlateau(tl_optim, 'min', factor=0.5, patience=10)
        # Training Modes
        task_learner.train()

        train_losses = []
        max_epochs = 50
        # max_runs = 3
        # for run in range(1, max_runs+1):
        for epoch in range(max_epochs):
            for sequences, lengths, tags in dataloader_l:

                if torch.cuda.is_available():
                    sequences = sequences.to(self.device)
                    lengths = lengths.to(self.device)
                    tags = tags.to(self.device)

                # Strip off tag padding and flatten
                # this occurs to the sequences of tokens in the forward pass of the RNNs
                # we do it here to match them and make loss computations faster
                tags = trim_padded_seqs(batch_lengths=lengths,
                                                batch_sequences=tags,
                                                pad_idx=self.pad_idx).view(-1)

                # Task Learner Step
                tl_optim.zero_grad()   # TODO: confirm if this gradient zeroing is correct
                tl_preds = task_learner(sequences, lengths)
                # print(tl_preds.shape)
                tl_loss = tl_loss_fn(tl_preds, tags)
                tl_loss.backward()
                tl_optim.step()
                # decay lr
                tl_sched.step(tl_loss)

                train_losses.append(tl_loss.item())
            average_train_loss = np.average(train_losses)
            self.tb_writer.add_scalar('Loss/TaskLearner/train', np.average(average_train_loss), epoch)

            # Get val metrics
            val_metrics = self.evaluation(task_learner=task_learner, dataloader=self.val_dataloader, task_type='NER')
            self.tb_writer.add_scalar('Metrics/TaskLearner/val/f1_macro', val_metrics[0]*100, epoch)
            self.tb_writer.add_scalar('Metrics/TaskLearner/val/f1_micro', val_metrics[1]*100, epoch)

            print(f'epoch {epoch} - ave loss {average_train_loss:0.3f} - Macro {val_metrics[0]*100:0.2f}% Micro {val_metrics[1]*100:0.2f}%')

        return self.evaluation(task_learner=task_learner, dataloader=self.test_dataloader, task_type='NER')

    def _random_sampling(self, dataloader_l, dataloader_u, dataloader_v, dataloader_t, unlabelled_indices):
        """ Performs active learning with IID random sampling

        Notes
        -----
        Random sampling sets the minimum acceptable performance in terms of
        model accuracy/f1 but also the ceiling on sampling/computational speed
        """

        metrics = self.train(dataloader_l=dataloader_l,
                            dataloader_u=dataloader_u,
                            dataloader_v=dataloader_v,
                            dataloader_t=dataloader_t,
                            mode=None)

        sampled_indices = self.sample_random(indices=unlabelled_indices)

        return metrics, sampled_indices

    def _least_confidence(self):
        """ Performs active learning with least confidence heuristic

        """
        pass


def main(config):
    exp = Experimenter(config)
    # Performs AL routine
    exp.learn()

    # Get full data performance
    # metrics = exp._full_data_performance()
    # print(f'F1 Macro {metrics[0]*100:0.2f}% Micro {metrics[1]*100:0.2f}%')

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