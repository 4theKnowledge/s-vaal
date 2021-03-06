"""
Trainer for generalisation of S-VAAL model.

TODO:
- Add model caching/saving
- Add model restart/checkpointing

- To access tensorboard run: tensorboard --logdir=runs

TODO:
    - investigate why we need to do this, likely as the task learner is stronger than the adversarial/svae?
    - Review adversarial hyperparameter for SVAE loss func
    - investigate why output labels are required for SVAE loss function...
    

@author: Tyler Bikaun
"""

import yaml
import numpy as np
import os
import unittest
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
Tensor = torch.Tensor

from models import TaskLearner, SVAE, Discriminator
from utils import to_var, trim_padded_seqs, load_json, split_data
from data import SequenceDataset, RealDataset
from connections import load_config

from pytorchtools import EarlyStopping


class Trainer:
    """ Prepares and trains S-VAAL model """
    def __init__(self):
        self.config = load_config()
        self.model_config = self.config['Models']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.task_type = self.config['Utils']['task_type']
        self.max_sequence_length = self.config['Utils'][self.task_type]['max_sequence_length']
        
        # Real data
        self.data_name = self.config['Utils'][self.task_type]['data_name']
        self.data_splits = self.config['Utils'][self.task_type]['data_split']
        self.pad_idx = self.config['Utils']['special_token2idx']['<PAD>']
        
        # Test run properties
        self.epochs = self.config['Train']['epochs']
        self.svae_iterations = self.config['Train']['svae_iterations']
        self.dsc_iterations = self.config['Train']['discriminator_iterations']
        self.adv_hyperparam = self.config['Models']['SVAE']['adversarial_hyperparameter']

    def _init_dataset(self, batch_size=None):
        """ Initialise real datasets by reading encoding data
        Returns
        -------
            self : dict
                Dictionary of DataLoaders
        Notes
        -----
        - Task type and data name are specified in the configuration file
        - Keys in 'data' are the splits used and the keys in 'vocab' are words and tags
        """
        
        kfold_xval = False
        
        self.x_y_pair_name = 'seq_label_pairs_enc' if self.data_name == 'ag_news' else 'seq_tags_pairs_enc' # Key in dataset - semantically correct for the task at hand.

        if batch_size is None:
            batch_size = self.config['Train']['batch_size']


        # Load pre-processed data
        path_data = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'data.json')
        path_vocab = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'vocabs.json')
        data = load_json(path_data)
        self.vocab = load_json(path_vocab)       # Required for decoding sequences for interpretations. TODO: Find suitable location... or leave be...
        self.vocab_size = len(self.vocab['words'])  # word vocab is used for model dimensionality setting + includes special characters (EOS, SOS< UNK, PAD)
        self.tagset_size = len(self.vocab['tags'])  # this includes special characters (EOS, SOS, UNK, PAD)

        self.datasets = dict()
        if kfold_xval:
            # Perform k-fold cross-validation
            # Join all datasets and then randomly assign train/val/test
            print('hello')
            
            
            for split in self.data_splits:
                print(data[split][self.x_y_pair_name])
            
            
        else:    
            for split in self.data_splits:
                # Access data
                split_data = data[split][self.x_y_pair_name]
                # Convert lists of encoded sequences into tensors and stack into one large tensor
                split_seqs = torch.stack([torch.tensor(enc_pair[0]) for key, enc_pair in split_data.items()])
                split_tags = torch.stack([torch.tensor(enc_pair[1]) for key, enc_pair in split_data.items()])
                # Create torch dataset from tensors
                split_dataset = RealDataset(sequences=split_seqs, tags=split_tags)
                # Add to dictionary
                self.datasets[split] = split_dataset #split_dataloader
                
                # Create torch dataloader generator from dataset
                if split == 'test':
                    self.test_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                if split == 'valid':
                    self.val_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f'{datetime.now()}: Data initialised succesfully')

    def _init_models(self, mode: str):
        """ Initialises models, loss functions, optimisers and sets models to training mode """
        # Task Learner
        self.task_learner = TaskLearner(**self.model_config['TaskLearner']['Parameters'],
                                        vocab_size=self.vocab_size,
                                        tagset_size=self.tagset_size,
                                        task_type=self.task_type).to(self.device)
        # Loss functions
        if self.task_type == 'SEQ':
            self.tl_loss_fn = nn.NLLLoss().to(self.device)
        if self.task_type == 'CLF':
            self.tl_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Optimisers
        self.tl_optim = optim.SGD(self.task_learner.parameters(),
                                  lr=self.model_config['TaskLearner']['learning_rate'])#, momentum=0, weight_decay=0.1)
        
        # Learning rate scheduler
        # Note: LR likely GT Adam
        # self.tl_sched = optim.lr_scheduler.ReduceLROnPlateau(self.tl_optim, 'min', factor=0.5, patience=10)
        # Training Modes
        self.task_learner.train()

        # SVAAL needs to initialise SVAE and DISC in addition to TL
        if mode == 'svaal':
            # Models
            self.svae = SVAE(**self.model_config['SVAE']['Parameters'],
                             vocab_size=self.vocab_size).to(self.device)
            self.discriminator = Discriminator(**self.model_config['Discriminator']['Parameters']).to(self.device)
            
            # Loss Function (SVAE defined within its class)
            self.dsc_loss_fn = nn.BCELoss().to(self.device)
            
            # Optimisers
            # Note: Adam will likely have a lower lr than SGD
            self.svae_optim = optim.Adam(self.svae.parameters(),
                                         lr=self.model_config['SVAE']['learning_rate'])
            self.dsc_optim = optim.Adam(self.discriminator.parameters(),
                                        lr=self.model_config['Discriminator']['learning_rate'])
            
            # Training Modes
            self.svae.train()
            self.discriminator.train()

        print(f'{datetime.now()}: Models initialised successfully')

    def train(self, dataloader_l, dataloader_u, dataloader_v, dataloader_t, mode: str, meta: str):
        """ 
        Sequentially train S-VAAL in the following training sequence:
            ```
                for epoch in epochs:
                    train Task Learner
                    for step in steps:
                        train SVAE
                    for step in steps:
                        train Discriminator
            ```
        Arguments
        ---------
            dataloader_l : TODO
                DataLoader for labelled data
            dataloader_u : TODO
                DataLoader for unlabelled data
            dataloader_v : TODO
                DataLoader for validation data
            mode : str
                Training mode (svaal, random, least_confidence, etc.)
            meta : str
                Meta data about the current training run

        Returns
        -------
            eval_metrics : tuple
                Task dependent evaluation metrics (F1 micro/macro or Accuracy)
            svae : TODO
                Sentence variational autoencoder
            discriminator : TODO
                Discriminator

        Notes
        -----

        """
        self.tb_writer = SummaryWriter(comment=meta, filename_suffix=meta)

        early_stopping = EarlyStopping(patience=self.config['Train']['es_patience'], verbose=True, path="checkpoints/checkpoint.pt")  # TODO: Set EarlyStopping params in config

        dataset_size = len(dataloader_l) + len(dataloader_u) if dataloader_u is not None else len(dataloader_l)
        train_iterations = dataset_size * (self.epochs+1)
        print(f'{datetime.now()}: Dataset size {dataset_size} Training iterations {train_iterations}')


        write_freq = 50 # number of iters to write to TensorBoard
        train_str = ''
        step = 0    # Used for KL annealing
        epoch = 1
        for train_iter in tqdm(range(train_iterations), desc='Training iteration'):            
            batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(dataloader_l))

            if torch.cuda.is_available():
                batch_sequences_l = batch_sequences_l.to(scurrent_indicesdataloader_u))
                batch_sequences_u = batch_sequences_u.to(self.device)
                batch_length_u = batch_lengths_u.to(self.device)

            # Strip off tag padding and flatten
            # Don't do sequences here as its done in the forward pass of the seq2seq models
            batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                            batch_sequences=batch_tags_l,
                                            pad_idx=self.pad_idx).view(-1)

            # Task Learner Step
            self.tl_optim.zero_grad()
            tl_preds = self.task_learner(batch_sequences_l, batch_lengths_l)
            tl_loss = self.tl_loss_fn(tl_preds, batch_tags_l)
            tl_loss.backward()
            self.tl_optim.step()
            
            if (train_iter > 0) & (train_iter % dataset_size == 0):
                # TODO: Reinstate LR scheduling in the future
                # Decay learning rate at the end of each epoch (if required)
                # self.tl_sched.step(tl_loss)     # Decay learning rate
                
                # Manually decay LR at each epoch
                # self.tl_optim.param_groups[0]["lr"] = self.tl_optim.param_groups[0]["lr"] / 10
                pass

            if mode == 'svaal':
                # Used in SVAE and Discriminator
                batch_size_l = batch_sequences_l.size(0)
                batch_size_u = batch_sequences_u.size(0)

                # SVAE Step
                # TODO: Extend for unsupervised - need to review svae.loss_fn for unsupervised case
                for i in range(self.svae_iterations):
                    # Labelled and unlabelled forward passes through SVAE and loss computation
                    logp_l, mean_l, logv_l, z_l = self.svae(batch_sequences_l, batch_lengths_l)
                    NLL_loss_l, KL_loss_l, KL_weight_l = self.svae.loss_fn(
                                                                    logp=logp_l,
                                                                    target=batch_sequences_l,
                                                                    length=batch_lengths_l,
                                                                    mean=mean_l,
                                                                    logv=logv_l,
                                                                    anneal_fn=self.model_config['SVAE']['anneal_function'],
                                                                    step=step,
                                                                    k=self.model_config['SVAE']['k'],
                                                                    x0=self.model_config['SVAE']['x0'])

                    logp_u, mean_u, logv_u, z_u = self.svae(batch_sequences_u, batch_lengths_u)
                    NLL_loss_u, KL_loss_u, KL_weight_u = self.svae.loss_fn(
                                                                    logp=logp_u,
                                                                    target=batch_sequences_u,
                                                                    length=batch_lengths_u,
                                                                    mean=mean_u,
                                                                    logv=logv_u,
                                                                    anneal_fn=self.model_config['SVAE']['anneal_function'],
                                                                    step=step,
                                                                    k=self.model_config['SVAE']['k'],
                                                                    x0=self.model_config['SVAE']['x0'])
                    # VAE loss
                    svae_loss_l = (NLL_loss_l + KL_weight_l * KL_loss_l) / batch_size_l
                    svae_loss_u = (NLL_loss_u + KL_weight_u * KL_loss_u) / batch_size_u

                    # Adversary loss - trying to fool the discriminator!
                    dsc_preds_l = self.discriminator(z_l)   # mean_l
                    dsc_preds_u = self.discriminator(z_u)   # mean_u
                    dsc_real_l = torch.ones(batch_size_l)
                    dsc_real_u = torch.ones(batch_size_u)

                    if torch.cuda.is_available():
                        dsc_real_l = dsc_real_l.to(self.device)
                        dsc_real_u = dsc_real_u.to(self.device)

                    # Higher loss = discriminator is having trouble figuring out the real vs fake
                    # Generator wants to maximise this loss
                    adv_dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)
                    adv_dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)
                    adv_dsc_loss = adv_dsc_loss_l + adv_dsc_loss_u

                    total_svae_loss = svae_loss_u + svae_loss_l + self.adv_hyperparam * adv_dsc_loss
                    self.svae_optim.zero_grad()
                    total_svae_loss.backward()
                    self.svae_optim.step()

                    # Add scalar for adversarial loss
                    # self.tb_writer.add_scalar('Loss/SVAE/train/labelled/ADV', NLL_loss_l, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/unabelled/ADV', NLL_loss_l, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/ADV_total', NLL_loss_l, i + (train_iter*self.svae_iterations))
                    # Add scalars for ELBO (NLL), KL divergence, and Total loss 
                    # self.tb_writer.add_scalar('Utils/SVAE/train/kl_weight_l', KL_weight_l, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Utils/SVAE/train/kl_weight_u', KL_weight_u, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/labelled/NLL', NLL_loss_l, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled/NLL', NLL_loss_u, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/labelled/KL_loss', KL_loss_l, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled/KL_loss', KL_loss_u, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/labelled/total', svae_loss_l, i + (train_iter*self.svae_iterations))
                    # self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled/total', svae_loss_u, i + (train_iter*self.svae_iterations))

                    # Sample new batch of data while training adversarial network
                    if i < self.svae_iterations - 1:
                        batch_sequences_l, batch_lengths_l, _ =  next(iter(dataloader_l))
                        batch_sequences_u, batch_length_u, _ = next(iter(dataloader_u))

                        if torch.cuda.is_available():
                            batch_sequences_l = batch_sequences_l.to(self.device)
                            batch_lengths_l = batch_lengths_l.to(self.device)
                            batch_sequences_u = batch_sequences_u.to(self.device)
                            batch_length_u = batch_length_u.to(self.device)
                        
                    # Increment step
                    step += 1

                # SVAE train_iter loss after iterative cycle
                self.tb_writer.add_scalar('Loss/SVAE/train/Total', total_svae_loss, train_iter)

                # Discriminator Step
                for j in range(self.dsc_iterations):

                    with torch.no_grad():
                        _, mean_l, _, z_l = self.svae(batch_sequences_l, batch_lengths_l)
                        _, mean_u, _, z_u = self.svae(batch_sequences_u, batch_lengths_u)

                    dsc_preds_l = self.discriminator(z_l)  #mean_l
                    dsc_preds_u = self.discriminator(z_u)  #mean_u

                    dsc_real_l = torch.ones(batch_size_l)
                    dsc_real_u = torch.zeros(batch_size_u)

                    if torch.cuda.is_available():
                        dsc_real_l = dsc_real_l.to(self.device)
                        dsc_real_u = dsc_real_u.to(self.device)

                    # Discriminator wants to minimise the loss here
                    dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)
                    dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)
                    total_dsc_loss = dsc_loss_l + dsc_loss_u
                    self.dsc_optim.zero_grad()
                    total_dsc_loss.backward()
                    self.dsc_optim.step()

                    # Sample new batch of data while training adversarial network
                    if j < self.dsc_iterations - 1:
                        # TODO: strip out unnecessary information
                        batch_sequences_l, batch_lengths_l, _ =  next(iter(dataloader_l))
                        batch_sequences_u, batch_length_u, _ = next(iter(dataloader_u))

                        if torch.cuda.is_available():
                            batch_sequences_l = batch_sequences_l.to(self.device)
                            batch_lengths_l = batch_lengths_l.to(self.device)
                            batch_sequences_u = batch_sequences_u.to(self.device)
                            batch_length_u = batch_length_u.to(self.device)
                        
                    # self.tb_writer.add_scalar('Loss/Discriminator/train/labelled', dsc_loss_l, i + (train_iter*self.dsc_iterations))
                    # self.tb_writer.add_scalar('Loss/Discriminator/train/unlabelled', dsc_loss_u, i + (train_iter*self.dsc_iterations))
                    # self.tb_writer.add_scalar('Loss/Discriminator/train', total_dsc_loss, i + (train_iter*self.dsc_iterations))

            if train_iter % write_freq == 0:
                self.tb_writer.add_scalar('Loss/TaskLearner/train', tl_loss, train_iter)

            # Wait until KL annealing has finished
            # ASsumes logistic, otherwise need to review 2* heuristic as x0 specifies midpoint of logistic function
            # Need to wait until x0 is reached to start early stopping 
            # Note: Removed conditional that early stopping gets initated if KL annealing is finished. THis issue with this is that after KL annealing the model doesn't have enough time to converge and stops to early unlike other methods.
            if (train_iter % dataset_size == 0):   # (step*2 >= self.model_config['SVAE']['x0']) or ((mode != 'svaal') & # for svall wait until KL annealing, for other models wait until firs epoch complete    
                print(f'{" KL Annealing! - Initiating Early Stopping" if mode == "svaal" else "Initiating Early Stopping"}')
                early_stopping(tl_loss, self.task_learner)      # TODO: Review. Should this be the metric we early stop on?
                
                if early_stopping.early_stop:
                    print(f'Early stopping at {train_iter}/{train_iterations} training iterations')
                    break

            if (train_iter > 0) & (epoch == 1 or train_iter % dataset_size == 0):
                # Tracks scalars for every iteration until one epoch is complete and then once an epoch.
                
                if (train_iter % write_freq == 0) or (train_iter % dataset_size == 0):
                    # Check accuracy/F1 of task learner on validation set
                    # Only compute when required, not every time.
                    val_metrics = self.evaluation(task_learner=self.task_learner,
                                                    dataloader=dataloader_v,
                                                    task_type=self.task_type)
                
                # Returns tuple if SEQ otherwise singular variable if CLF
                if train_iter % dataset_size == 0:
                    val_string = f'Task Learner ({self.task_type}) Validation ' + f'Scores:\nF1: Macro {val_metrics["f1 macro"]*100:0.2f}% Micro {val_metrics["f1 micro"]*100:0.2f}%' if self.task_type == 'SEQ' else f'Accuracy {val_metrics["accuracy"]*100:0.2f}'
                    train_str += val_string + '\n'
                    print(val_string)

                if train_iter % write_freq == 0:
                    if self.task_type == 'SEQ':
                        self.tb_writer.add_scalar('Metrics/TaskLearner/val/f1_macro', val_metrics["f1 macro"]*100, train_iter)
                        self.tb_writer.add_scalar('Metrics/TaskLearner/val/f1_micro', val_metrics["f1 micro"]*100, train_iter)
                        # self.tb_writer.add_scalar('Metrics/TaskLearner/val/precision_macro', val_metrics["precision macro"]*100, train_iter)
                        # self.tb_writer.add_scalar('Metrics/TaskLearner/val/precision_micro', val_metrics["precision micro"]*100, train_iter)
                        # self.tb_writer.add_scalar('Metrics/TaskLearner/val/recall_macro', val_metrics["recall macro"]*100, train_iter)
                        # self.tb_writer.add_scalar('Metrics/TaskLearner/val/recall_micro', val_metrics["recall micro"]*100, train_iter)

                    if self.task_type == 'CLF':
                        self.tb_writer.add_scalar('Metrics/TaskLearner/val/acc', val_metrics["accuracy"], train_iter)

            # Computes each epoch (full data pass)
            if (train_iter > 0) & (train_iter % dataset_size == 0):
                # TODO: Add test evaluation metric scalar for tb here!! Currently only getting validation

                if mode == 'svaal':
                    train_iter_str = f'Train Iter {train_iter} - Losses (TL-{self.task_type} {tl_loss:0.2f} | SVAE {total_svae_loss:0.2f} | Disc {total_dsc_loss:0.2f} | Learning rates: TL ({self.tl_optim.param_groups[0]["lr"]})'
                else:
                    train_iter_str = f'Train Iter {train_iter} - Losses (TL-{self.task_type} {tl_loss:0.2f}) | Learning rate ({self.tl_optim.param_groups[0]["lr"]:0.2e})'
                train_str += train_iter_str + '\n'
                print(train_iter_str)

                # Completed an epoch
                print(f'Completed epoch: {epoch}')
                epoch += 1

        # Compute test metrics
        test_metrics_final = self.evaluation(task_learner=self.task_learner,
                                             dataloader=dataloader_t,
                                             task_type='SEQ')

        if mode == 'svaal':
            return test_metrics_final, self.svae, self.discriminator
        else:
            # This may be updated to return the TL for non-adversarial heuristics
            return test_metrics_final

    def evaluation(self, task_learner, dataloader, task_type):
        """ Computes performance metrics on holdout sets (val, train) for the task learner
        
        Arguments
        ---------
            task_learner : TODO
                TODO
            dataloader : TODO
                TODO
            task_type : str
                Type of model task e.g. CLF or SEQ
        
        Returns
        -------
            metric : float
                Accuracy (CLF) or F1 scores (SEQ)
        """

        task_learner.eval()
        preds_all = []
        true_labels_all = []
        for batch_sequences, batch_lengths, batch_labels in dataloader:
            if torch.cuda.is_available():
                batch_sequences = batch_sequences.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                batch_labels = batch_labels.to(self.device)
            with torch.no_grad():
                preds = task_learner(batch_sequences, batch_lengths)
            # Get argmax of preds
            preds_argmax = torch.argmax(preds, dim=1)
            # print(f'preds:{preds.shape} - preds_argmax:{preds_argmax.shape} - batch_labels: {batch_labels.view(-1).shape}')
            preds_all.append(preds_argmax)
            true_labels_all.append(batch_labels.view(-1))   # need to convert batch_labels dims: (batch_size, tagset_size) -> (batch_size * tagset_size)
        preds_all = torch.cat(preds_all, dim=0)
        true_labels_all = torch.cat(true_labels_all, dim=0)
        # Need to reset task_learner back to train mode.
        task_learner.train()
        if task_type == 'SEQ':
            f1_macro = f1_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='macro')
            f1_micro = f1_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='micro')
            # p_macro = precision_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='macro')
            # p_micro = precision_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='micro')
            # r_macro = recall_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='macro')
            # r_micro = recall_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='micro')
            
            return {"f1 macro": f1_macro,
                    "f1 micro": f1_micro}
            
            # return {"f1 macro": f1_macro,
            #         "f1 micro":f1_micro,
            #         "precision macro": p_macro,
            #         "precision micro": p_micro,
            #         "recall macro": r_macro,
            #         "recall micro": r_micro}

        if task_type == 'CLF':
            # TODO: Add precision and recall metrics
            acc = accuracy_score(y_true=true_labels_all.cpu().nump(), y_pred=preds_all.cpu().numpy())
            return {"accuracy": acc}


def main():
    # Train S-VAAL model
    trainer = Trainer()
    trainer._init_dataset()
    # trainer._init_models()
    # trainer.train()

if __name__ == '__main__':
    # Seeds
    # config = load_config()
    # np.random.seed(config['Train']['seed'])
    # torch.manual_seed(config['Train']['seed'])

    main()