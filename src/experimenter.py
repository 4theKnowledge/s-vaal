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
import sys, traceback
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from train import Trainer
from sampler import Sampler
from models import TaskLearner, SVAE
from utils import trim_padded_seqs
from connections import load_config, Mongo
from pytorchtools import EarlyStopping

# TODO: Write decorator that wraps around Trial Runner and saves data into mongo db under collection 'experiments'

class TrialRunner(object):
    """ Decorator for running n trials of a function """
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
        
        start_time = datetime.now()
        run_stats = dict()
        if self.model_name == 'svaal':
            run_samples = dict()
            run_preds = dict()
            run_all_pred_stats = dict()
            for run in range(1, self.runs+1):
                result, samples, preds, all_pred_stats = func()
                # print(f'Result of run {run}: {result}')
                run_stats[str(run)] = result
                run_samples[str(run)] = samples
                run_preds[str(run)] = preds
                run_all_pred_stats[str(run)] = all_pred_stats

            finish_time = datetime.now()
            run_time = (finish_time-start_time).total_seconds()/60
            return run_stats, run_samples, run_preds, run_all_pred_stats, start_time, finish_time, run_time
        else:
            for run in range(1, self.runs+1):
                result = func()
                print(f'Result of run {run}: {result}')
                run_stats[str(run)] = result

            finish_time = datetime.now()
            run_time = (finish_time-start_time).total_seconds()/60
            return run_stats, start_time, finish_time, run_time

class Experimenter(Trainer, Sampler):
    def __init__(self):
        Trainer.__init__(self)
        config = load_config()
        self.config = config

        self.initial_budget_frac = config['Train']['init_budget_frac']
        self.budget_frac = config['Train']['budget_frac']
        self.data_splits_frac = np.round(np.linspace(self.budget_frac, self.budget_frac*10, num=10, endpoint=True), 2)
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
        self.budget = math.ceil(self.budget_frac*dataset_size)  # currently can only have a fixed budget size
        Sampler.__init__(self, self.budget)

        all_indices = set(np.arange(dataset_size))
        k_initial = math.ceil(len(all_indices)*self.initial_budget_frac)
        initial_indices = random.sample(list(all_indices), k=k_initial)

        sampler_init = data.sampler.SubsetRandomSampler(initial_indices)    # need to sample from training dataset

        self.labelled_dataloader = data.DataLoader(train_dataset, sampler=sampler_init, batch_size=self.batch_size, drop_last=True)
        self.val_dataloader = data.DataLoader(self.datasets['valid'], batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = data.DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=True, drop_last=False)

        print(f'{datetime.now()}: Data initialised - Sizes: Train {len(self.labelled_dataloader)} Valid {len(self.val_dataloader)} Test {len(self.test_dataloader)}')
        
        return all_indices, initial_indices

    def learn(self):
        """ Performs active learning routine"""

        # Bookkeeping
        results = dict()    # Result metrics
        samples = dict()    # Sampled indices
        disc_preds = dict() # Discriminator predictions (only of those selected for sampling)
        disc_all_preds_stats = dict()   # Statistics on all predictions made by discriminator

        print(f'RUNNING MODEL UNDER {self.data_splits_frac} SPLIT REGIME')
        
        all_indices, initial_indices = self._init_al_data()
        current_indices = list(initial_indices)
        
        for split in self.data_splits_frac:
            print(f'\nSPLIT - {split*100:0.0f}%')
            meta = f' {self.al_mode} run x data split {split*100:0.0f}'

            # Initialise models for retraining
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
                print("Breaking at 100% of data - can't run SVAAL with no unlabelled data atm")
                break

            # Perform AL training and sampling
            if self.al_mode == 'svaal':
                metrics, sampled_indices, preds, all_pred_stats = self._svaal(self.labelled_dataloader,
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

            print(f'Test Eval.: F1 Scores - Macro {metrics["f1 macro"]*100:0.2f}% Micro {metrics["f1 micro"]*100:0.2f}%')        
            
            # Record indices of labelled samples before sampling
            # Note: Need to convert indices into int dtype as they are np.int64 which mongo doesn't understand
            samples[str(int(split*100))] = {'labelled': [int(index) for index in current_indices]}
            
            # Add new samples to labelled dataset
            current_indices = list(current_indices) + list(sampled_indices)
            sampler = data.sampler.SubsetRandomSampler(current_indices)
            self.labelled_dataloader = data.DataLoader(self.datasets['train'], sampler=sampler, batch_size=self.batch_size, drop_last=True)

            # Record performance at each split
            results[str(int(split*100))] = metrics

            # Record predictions made by the discriminator
            if self.al_mode == 'svaal':
                disc_preds[str(int(split*100))] = preds.cpu().tolist()
                disc_all_preds_stats[str(int(split*100))] = all_pred_stats


        # Increment run number and reset if at the end of trial
        self.run_no += 1
        self._check_reset_run_no()
        if self.al_mode == 'svaal':
            return results, samples, disc_preds, disc_all_preds_stats
        else:
            return results, samples
        
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
        sampled_indices, preds, all_pred_stats = self.sample_adversarial(svae=svae,
                                                            discriminator=discriminator,
                                                            data=dataloader_u,
                                                            indices=unlabelled_indices,
                                                            cuda=True)    # TODO: review usage of indices arg
        return metrics, sampled_indices, preds, all_pred_stats

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

        # If not running optimisation and passing parameters through model... set to defaults in config
        if parameterisation is None:
            parameterisation = {"batch_size": self.config['Train']['batch_size']}
            parameterisation.update({"epochs": self.config['Train']['epochs']})
            parameterisation.update(self.config['Models']['TaskLearner']['Parameters'])
            parameterisation.update({"learning_rate": self.config['Models']['TaskLearner']['learning_rate']})

        self._init_dataset(batch_size=parameterisation["batch_size"])
        
        dataloader_l = data.DataLoader(dataset=self.datasets['train'],
                                        batch_size=parameterisation["batch_size"],
                                        shuffle=True,
                                        num_workers=0)

        params = {"embedding_dim": parameterisation["tl_embedding_dim"],
                    "hidden_dim": parameterisation["tl_hidden_dim"]}
        
        # Initialise model
        task_learner = TaskLearner(**params,
                                    vocab_size=self.vocab_size,
                                    tagset_size=self.tagset_size,
                                    task_type=self.task_type).to(self.device)
        if self.task_type == 'SEQ':
            tl_loss_fn = nn.NLLLoss().to(self.device)
            
        if self.task_type == 'CLF':
            tl_loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        
        tl_optim = optim.SGD(task_learner.parameters(), lr=parameterisation["learning_rate"], momentum=0)       # TODO: Update with params from config
        # tl_sched = optim.lr_scheduler.ReduceLROnPlateau(tl_optim, 'min', factor=self.config['Train']['lr_sched_factor'], patience=self.config['Train']['lr_patience'])
        early_stopping = EarlyStopping(patience=self.config['Train']['es_patience'], verbose=False, path="checkpoints/checkpoint.pt")  # TODO: Set EarlyStopping params in config
        task_learner.train()

        dataset_size = len(dataloader_l)    # no. batches
        train_iterations = dataset_size * (parameterisation["epochs"]+1)
        
        print(f'{datetime.now()}: Dataset size {dataset_size} - Training iterations {train_iterations}')
        train_losses = []
        train_val_metrics = []
        epoch = 1
        for train_iter in tqdm(range(train_iterations), desc="Training iteration"):
            sequences, lengths, tags = next(iter(dataloader_l))
            
            if torch.cuda.is_available():
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                tags = tags.to(self.device)

            tags = trim_padded_seqs(batch_lengths=lengths,
                                    batch_sequences=tags,
                                    pad_idx=self.pad_idx).view(-1)

            # Task Learner Step
            tl_optim.zero_grad()
            tl_preds = task_learner(sequences, lengths)
            # print(tl_preds.shape)
            tl_loss = tl_loss_fn(tl_preds, tags)
            tl_loss.backward()
            tl_optim.step()
            
            train_losses.append(tl_loss.item())

            if (train_iter > 0) & (train_iter % dataset_size == 0):
                # Evaluate as epoch complete
                
                # decay lr after each epoch    
                # tl_sched.step(tl_loss)

                average_train_loss = np.average(train_losses)
                tb_writer.add_scalar('Loss/TaskLearner/train', np.average(average_train_loss), train_iter)
                train_losses = []   # reset train losses for next epoch
                
                # Get val metrics
                val_metrics = self.evaluation(task_learner=task_learner, dataloader=self.val_dataloader, task_type='SEQ')
                train_val_metrics.append(val_metrics["f1 macro"])
                tb_writer.add_scalar('Metrics/TaskLearner/val/f1_macro', val_metrics["f1 macro"]*100, train_iter)
                tb_writer.add_scalar('Metrics/TaskLearner/val/f1_micro', val_metrics["f1 micro"]*100, train_iter)

                print(f'epoch {epoch} - ave loss {average_train_loss:0.3f} - Macro {val_metrics["f1 macro"]*100:0.2f}% Micro {val_metrics["f1 micro"]*100:0.2f}% LR {tl_optim.param_groups[0]["lr"]:0.2e}')
                
                early_stopping(val_metrics["f1 macro"], task_learner) # tl_loss        # Stopping on macro F1
                if early_stopping.early_stop:
                    print('Early stopping')
                    break

                epoch += 1

        average_val_metric = np.average(train_val_metrics)
        print(f'Average Validation - F1 Macro {average_val_metric*100:0.2f}%')

        # Test performance
        test_metrics = self.evaluation(task_learner=task_learner, dataloader=self.test_dataloader, task_type='SEQ')
        print(f'Test F1 Macro - {test_metrics["f1 macro"]*100:0.2f}')
        
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


        if parameterisation is None:
            parameterisation = {"batch_size": self.config['Train']['batch_size']}
            parameterisation.update(self.config['Models']['SVAE']['Parameters'])
            parameterisation.update({'epochs': self.config['Train']['epochs'], 
                                    "svae_k": self.config['Models']['SVAE']['k'],
                                    "svae_x0": self.config['Models']['SVAE']['x0']})


        self._init_dataset(batch_size=parameterisation["batch_size"])


        dataloader_l = data.DataLoader(dataset=self.datasets['train'],
                                        batch_size=parameterisation["batch_size"],
                                        shuffle=True,
                                        num_workers=0)

        params = {'embedding_dim': parameterisation['svae_embedding_dim'],
                    'hidden_dim': parameterisation['svae_hidden_dim'],
                    'rnn_type': parameterisation['svae_rnn_type'],
                    'num_layers': parameterisation['svae_num_layers'],
                    'bidirectional': parameterisation['svae_bidirectional'],
                    'latent_size': parameterisation['svae_latent_size'],
                    'word_dropout': parameterisation['svae_word_dropout'],
                    'embedding_dropout': parameterisation['svae_embedding_dropout']}

        # Initialise model
        
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
                                                            k=parameterisation['svae_k'], #self.config['Models']['SVAE']['k'],
                                                            x0=parameterisation['svae_x0']) #self.config['Models']['SVAE']['x0'])
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
    """ Runs task learner and svae individually """
    exp = Experimenter()

    @TrialRunner(runs=exp.config['Train']['max_runs'], model_name='FDP')
    def run_fdp():
        output_metric = exp._full_data_performance()
        return output_metric
    run_stats_fdp, _, _, _ = run_fdp
    print(f'FDP results:\n{run_stats_fdp}')

    # @TrialRunner(runs=exp.config['Train']['max_runs'], model_name='SVAE (zdim 64 - hedim 512)')
    # def run_svae():
    #     output_metric = exp._svae()
    #     return output_metric
    # run_stats_svae, _, _, _ = run_svae
    # print(f'SVAE results:\n{run_stats_svae}')


def run_al():
    """ Runs active learning routine """
    exp = Experimenter()
    mongo_coll_conn = Mongo(collection_name='experiments')

    @TrialRunner(runs=exp.config['Train']['max_runs'], model_name='svaal')
    def run_svaal():
        output_metric, samples, preds, run_all_pred_stats = exp.learn()     # TODO: Give a more meaniningful name...
        return output_metric, samples, preds, run_all_pred_stats
    run_stats_svaal, run_samples, run_preds, run_all_pred_stats, start_time, finish_time, run_time = run_svaal

    # print(f'SVAE results:\n{run_stats_svaal}')

    data = {"name": "svaal",
            "info": {"start timestamp": start_time,
                        "finish timestamp": finish_time,
                        "run time": run_time},
            "settings": exp.config,
            "results": run_stats_svaal,
            "samples": run_samples,
            "predictions": {"results": run_preds, 
                            "statistics": run_all_pred_stats}
            }

    # Post results to mongodb
    mongo_coll_conn.post(data)
    
def run_random():
    """ Runs random sampling in active learning fashion """
    exp = Experimenter()
    mongo_coll_conn = Mongo(collection_name='experiments')

    @TrialRunner(runs=exp.config['Train']['max_runs'], model_name='random')
    def run_random():
        output_metric = exp.learn()     # TODO: Give a more meaniningful name...
        return output_metric
    run_stats_random, start_time, finish_time, run_time = run_random

    data = {"name": "random",
            "info": {"start timestamp": start_time,
                        "finish timestamp": finish_time,
                        "run time": run_time},
            "settings": exp.config,
            "results": run_stats_random
            }
    
    # Post results to mongodb
    mongo_coll_conn.post(data)


if __name__ == '__main__':
    # Seeds
    # config = load_config()
    # np.random.seed(config['Train']['seed'])
    # torch.manual_seed(config['Train']['seed'])

    # run_individual_models()
    # run_random()
    run_al()