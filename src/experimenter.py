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
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from train import Trainer
from sampler import Sampler
from models import TaskLearner, SVAE
from utils import trim_padded_seqs
from connections import load_config
from pytorchtools import EarlyStopping


class TrialRunner(object):
    """ Decorator for running n trials of function """
    def __init__(self, runs=5, model_name=None):
        config = load_config()
        if config:
            self.runs = config['Train']['max_runs']
        else:
            self.runs = runs
        self.model_name = model_name

    def __call__(self, func):
        """ Performs n runs of function and returns dictionary of results """
        print(f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}: Running {self.model_name} {self.runs} times')
        
        run_stats = dict()
        for run in range(1, self.runs+1):
            result = func()
            print(f'Result of run {run}: {result:0.2f}')
            run_stats[str(run)] = result
        return run_stats


class DataSizer(object):
    def __init__(self):
        pass
    def __call__(self):
        pass



class Experimenter(Trainer, Sampler):
    def __init__(self):
        Trainer.__init__(self)
        config = load_config()
        self.config = config

        self.initial_budget_frac = config['Train']['init_budget_frac']
        self.budget_frac = config['Train']['budget_frac']
        self.data_splits_frac = np.round(np.linspace(self.budget_frac, 1, num=10, endpoint=True), 1)
        self.batch_size = config['Train']['batch_size']
        self.max_runs = config['Train']['max_runs']
        self.al_mode = config['Train']['al_mode']
        self.run_no = 1 # tracker for running models over n trials (TODO: ensure that this is robust and doesn't index wildly)

    def _check_reset_run_no(self):
        """
        Reset run number to 1 if it enumerates over the maximum number of permissible runs
        """
        if self.run_no == self.config['Train']['max_runs']+1:
            self.run_no = 1

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
        Sampler.__init__(self, self.budget)

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
        """ Performs active learning routine"""

        metrics_hist = dict()
        for run in range(1, self.max_runs+1):
            metrics_hist[str(run)] = dict()
            all_indices, initial_indices = self._init_al_data()
            current_indices = list(initial_indices)
            for split in self.data_splits_frac:
                print(f'\nRUN {run} - SPLIT - {split*100:0.0f}%')
                meta = f' {self.al_mode} run {run} data split {split*100:0.0f}' # Meta data for tb scalars

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
                                                        batch_size=self.config['Train']['batch_size'],
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
                                                            unlabelled_indices,
                                                            meta)
                elif self.al_mode == 'random':
                    metrics, sampled_indices = self._random_sampling(self.labelled_dataloader,
                                                                        unlabelled_dataloader,
                                                                        self.val_dataloader,
                                                                        self.test_dataloader,
                                                                        unlabelled_indices,
                                                                        meta)

                print(f'Test Eval.: F1 Scores - Macro {metrics[0]*100:0.2f}% Micro {metrics[1]*100:0.2f}%')        
                
                # Add new samples to labelled dataset
                current_indices = list(current_indices) + list(sampled_indices)
                sampler = data.sampler.SubsetRandomSampler(current_indices)
                self.labelled_dataloader = data.DataLoader(self.datasets['train'], sampler=sampler, batch_size=self.batch_size, drop_last=True)

                # Record performance at each split
                metrics_hist[str(run)][str(split)] = metrics

        # write results to disk
        with open('results.json', 'w') as fj:
            json.dump(metrics_hist, fj, indent=4)

    def _svaal(self, dataloader_l, dataloader_u, dataloader_v, dataloader_t, unlabelled_indices, meta):
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
                                                    mode='svaal',
                                                    meta=meta)
        sampled_indices = self.sample_adversarial(svae=svae,
                                                    discriminator=discriminator,
                                                    data=dataloader_u,
                                                    indices=unlabelled_indices,
                                                    cuda=True)    # TODO: review usage of indices arg
        return metrics, sampled_indices

    def _random_sampling(self, dataloader_l, dataloader_u, dataloader_v, dataloader_t, unlabelled_indices, meta):
        """ Performs active learning with IID random sampling

        Notes
        -----
        Random sampling sets the minimum expected performance in terms of
        model accuracy/f1 but also the ceiling on sampling/computational speed
        """

        metrics = self.train(dataloader_l=dataloader_l,
                                dataloader_u=dataloader_u,
                                dataloader_v=dataloader_v,
                                dataloader_t=dataloader_t,
                                mode=None,
                                meta=meta)

        sampled_indices = self.sample_random(indices=unlabelled_indices)

        return metrics, sampled_indices

    def _least_confidence(self, model, dataloader_l, dataloader_u, dataloader_v, dataloader_t, unlabelled_indices, meta):
        """ Performs active learning with least confidence heuristic"""
        
        metrics = self.train(dataloader_l=dataloader_l,
                                dataloader_u=dataloader_u,
                                dataloader_v=dataloader_v,
                                dataloader_t=dataloader_t,
                                mode=None,
                                meta=meta)

        sampled_indices = self.sample_least_confidence(model=task_learner,
                                                        data=dataloader_u,
                                                        indices=unlabelled_indices)

        return metrics, sampled_indices


    def _full_data_performance(self, parameterisation=None):
        """ Gets performance of task learner on full dataset without any active learning

        Arguments
        ---------
            parameterisation : dict
                Dictionary of parameters for task learner (batch size, embedding dim and hidden dim)

        Notes
        -----
        Parameterisation is passed in so that the function can be used with Bayesian optimisation routines.
        """

        tb_writer = SummaryWriter(comment=f'FDP run {self.run_no}', filename_suffix=f'FDP run {self.run_no}')
        self._init_dataset()

        if parameterisation is None:
            parameterisation = {"batch_size": self.config['Train']['batch_size']}
            parameterisation.update(self.config['Models']['TaskLearner']['Parameters'])

        dataloader_l = data.DataLoader(dataset=self.datasets['train'],
                                        batch_size=parameterisation["batch_size"],
                                        shuffle=True,
                                        num_workers=0)
        params = {"embedding_dim": parameterisation["embedding_dim"],
                    "hidden_dim": parameterisation["hidden_dim"]}
        # Initialise model
        task_learner = TaskLearner(**params,
                                    vocab_size=self.vocab_size,
                                    tagset_size=self.tagset_size,
                                    task_type=self.task_type).to(self.device)
        if self.task_type == 'SEQ':
            tl_loss_fn = nn.NLLLoss().to(self.device)
        if self.task_type == 'CLF':
            tl_loss_fn = nn.CrossEntropyLoss().to(self.device)
        tl_optim = optim.SGD(task_learner.parameters(), lr=self.config['Models']['TaskLearner']['learning_rate'], momentum=0)       # TODO: Update with params from config
        tl_sched = optim.lr_scheduler.ReduceLROnPlateau(tl_optim, 'min', factor=self.config['Train']['lr_sched_factor'], patience=self.config['Train']['lr_patience'])
        early_stopping = EarlyStopping(patience=self.config['Train']['es_patience'], verbose=False, path="checkpoints/checkpoint.pt")  # TODO: Set EarlyStopping params in config
        task_learner.train()

        train_losses = []
        train_val_metrics = []

        for epoch in range(1, self.config['Train']['epochs']+1):
            for sequences, lengths, tags in dataloader_l:

                if torch.cuda.is_available():
                    sequences = sequences.to(self.device)
                    lengths = lengths.to(self.device)
                    tags = tags.to(self.device)

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
            tb_writer.add_scalar('Loss/TaskLearner/train', np.average(average_train_loss), epoch)

            # Get val metrics
            val_metrics = self.evaluation(task_learner=task_learner, dataloader=self.val_dataloader, task_type='SEQ')
            train_val_metrics.append(val_metrics[0])
            tb_writer.add_scalar('Metrics/TaskLearner/val/f1_macro', val_metrics[0]*100, epoch)
            tb_writer.add_scalar('Metrics/TaskLearner/val/f1_micro', val_metrics[1]*100, epoch)

            print(f'epoch {epoch} - ave loss {average_train_loss:0.3f} - Macro {val_metrics[0]*100:0.2f}% Micro {val_metrics[1]*100:0.2f}%')

            early_stopping(val_metrics[0], task_learner) # tl_loss        # Stopping on macro F1
            if early_stopping.early_stop:
                print('Early stopping')
                break

        average_val_metric = np.average(train_val_metrics)
        print(f'Average Validation - F1 Macro {average_val_metric*100:0.2f}%')

        # Test performance
        test_metrics = self.evaluation(task_learner=task_learner, dataloader=self.test_dataloader, task_type='SEQ')

        # run_stats[str(run)] = {'Val Ave': average_val_metric*100, 'Test': test_metrics[0]*100}

        # return run_stats
        
        # Increment run number and reset if at the end of trial
        self.run_no += 1
        self._check_reset_run_no()
        return average_val_metric

    def _svae(self, parameterisation=None):
        """ Trains the SVAE for n runs

        Arguments
        ---------
            parameterisation : dict
                Dictionary of parameters relating to the svae model
        Returns
        -------
            svae_loss : float
                Loss of SVAE model after training
        Notes
        -----

        """
        tb_writer = SummaryWriter(comment=f' SVAE run {self.run_no}', filename_suffix=f' SVAE run {self.run_no}')
        self._init_dataset()

        if parameterisation is None:
            parameterisation = {"batch_size": self.config['Train']['batch_size']}
            parameterisation.update(self.config['Models']['SVAE']['Parameters'])
            parameterisation.update({'epochs': self.config['Train']['epochs']})

        dataloader_l = data.DataLoader(dataset=self.datasets['train'],
                                        batch_size=parameterisation["batch_size"],
                                        shuffle=True,
                                        num_workers=0)

        params = {'embedding_dim': parameterisation['embedding_dim'],
                    'hidden_dim': parameterisation['hidden_dim'],
                    'rnn_type': parameterisation['rnn_type'],
                    'num_layers': parameterisation['num_layers'],
                    'bidirectional': parameterisation['bidirectional'],
                    'latent_size': parameterisation['latent_size'],
                    'word_dropout': parameterisation['word_dropout'],
                    'embedding_dropout': parameterisation['embedding_dropout']}

        # Initialise model
        # Note: loss function is defined in SVAE class
        svae = SVAE(**params, vocab_size=self.vocab_size).to(self.device)
        
        svae_optim = optim.Adam(svae.parameters(), lr=self.config['Models']['SVAE']['learning_rate'])
        svae_sched = optim.lr_scheduler.ReduceLROnPlateau(svae_optim, 'min', factor=self.config['Train']['lr_sched_factor'], patience=self.config['Train']['lr_patience'])
        early_stopping = EarlyStopping(patience=self.config['Train']['es_patience'], verbose=False, path="checkpoints/checkpoint.pt")  # TODO: Set EarlyStopping params in config
        svae.train()

        train_losses = []
        step = 0
        for epoch in range(1, parameterisation['epochs']+1):
            for sequences, lengths, tags in dataloader_l:

                batch_size = len(sequences)     # Calculate batch size here as it can change e.g. if you're on the last batch.

                if torch.cuda.is_available():
                    sequences = sequences.to(self.device)
                    lengths = lengths.to(self.device)
                    tags = tags.to(self.device)

                tags = trim_padded_seqs(batch_lengths=lengths,
                                        batch_sequences=tags,
                                        pad_idx=self.pad_idx).view(-1)

                # SVAE Step
                svae_optim.zero_grad()
                logp, mean, logv, z = svae(sequences, lengths)
                
                NLL_loss, KL_loss, KL_weight = svae.loss_fn(logp=logp,
                                                            target=sequences,
                                                            length=lengths,
                                                            mean=mean,
                                                            logv=logv,
                                                            anneal_fn = self.config['Models']['SVAE']['anneal_function'],
                                                            step=step,
                                                            k=self.config['Models']['SVAE']['k'],
                                                            x0=self.config['Models']['SVAE']['x0'])
                svae_loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                svae_loss.backward()
                svae_optim.step()
                svae_sched.step(svae_loss)  # decay learning rate
                train_losses.append(svae_loss.item())
                
                # Add scalars
                tb_writer.add_scalar('Utils/SVAE/KL_weight', KL_weight, step)
                tb_writer.add_scalar('Loss/SVAE/train/NLL', NLL_loss.item() / batch_size, step)
                tb_writer.add_scalar('Loss/SVAE/train/KL', KL_loss.item() / batch_size, step)
                
                step += 1   # Step after each backwards pass through the network


            average_train_loss = np.average(train_losses)
            tb_writer.add_scalar('Loss/SVAE/train', np.average(average_train_loss), epoch)

            print(f'epoch {epoch} - ave loss {average_train_loss:0.3f}')

            early_stopping(average_train_loss, svae)
            if early_stopping.early_stop:
                print('Early stopping')
                break
                
            
        self.run_no += 1
        self._check_reset_run_no()
        return average_train_loss


def run_individual_models():
    """
    Runs task learner and svae individually.
    """
    exp = Experimenter()

    # @TrialRunner(runs=exp.config['Train']['max_runs'], model_name='FDP')
    # def run_fdp():
    #     output_metric = exp._full_data_performance()
    #     return output_metric
    # run_stats_fpd = run_fdp
    # print(f'FDP results:\n{run_stats_fpd}')

    @TrialRunner(runs=exp.config['Train']['max_runs'], model_name='SVAE (zdim 64 - hedim 512)')
    def run_svae():
        output_metric = exp._svae()
        return output_metric
    run_stats_svae = run_svae
    print(f'SVAE results:\n{run_stats_svae}')


def run_al():
    """
    Runs active learning routine
    """
    exp = Experimenter()
    # Performs AL routine
    exp.learn()

if __name__ == '__main__':
    # Seeds
    # config = load_config()
    # np.random.seed(config['Train']['seed'])
    # torch.manual_seed(config['Train']['seed'])

    run_individual_models()