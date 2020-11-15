"""
Trainer for generalisation of S-VAAL model.

TODO:
- Add tensorboard logging
- Add model caching/saving
- Add model restart/checkpointing

- To access tensorboard run: tensorboard --logdir=runs

@author: Tyler Bikaun
"""

import yaml
import numpy as np
import os
import unittest
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
Tensor = torch.Tensor

from tasklearner import TaskLearner  # Change import alias if using both models.
from models import SVAE, Discriminator
from utils import to_var, trim_padded_seqs, load_json, split_data
from data_generator import DataGenerator, SequenceDataset, RealDataset


class Trainer(DataGenerator):
    """ Prepares and trains S-VAAL model """
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        DataGenerator.__init__(self, config)

        self.config = config
        self.model_config = config['Model']

        self.tb_writer = SummaryWriter()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.task_type = self.config['Utils']['task_type']

        # Testing data properties
        self.batch_size = config['Train']['batch_size']
        self.max_sequence_length = config['Tester']['max_sequence_length']
        
        # Real data
        self.data_name = config['Utils'][self.task_type]['data_name']
        self.data_config = config['Data']
        self.data_splits = self.config['Utils'][self.task_type]['data_split']
        
        # Test run properties
        self.epochs = config['Train']['epochs']
        self.svae_iterations = config['Train']['svae_iterations']
        self.dsc_iterations = config['Train']['discriminator_iterations']
        self.learning_rates = config['Train']['learning_rates']
        self.adv_hyperparam = config['Train']['adversarial_hyperparameter']

        # Exe
        # self._init_dataset_gen()
        self._init_dataset()
        self._init_models()
        self.train()

    def _init_dataset_gen(self):
        """ Initialises dataset for model training """
        # Currently will be using generated data, but in the future will be real.

        self.train_dataset_l = SequenceDataset(self.config, no_sequences=8, max_sequence_length=self.max_sequence_length, task_type=self.task_type)
        self.train_dataloader_l = DataLoader(self.train_dataset_l, batch_size=2, shuffle=True, num_workers=0)

        self.train_dataset_u = SequenceDataset(self.config, no_sequences=16, max_sequence_length=self.max_sequence_length, task_type=self.task_type)
        self.train_dataloader_u = DataLoader(self.train_dataset_u, batch_size=2, shuffle=True, num_workers=0)

        # Concatenate sequences in X_l and X_u to build vocabulary for downstream
        self.vocab = self.build_vocab(sequences = torch.cat((self.train_dataset_l.sequences, self.train_dataset_u.sequences)))
        self.vocab_size = len(self.vocab)

        print('---- DATA SUCCESSFULLY INITIALISED ----')

    def _init_dataset(self):
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
        self.x_y_pair_name = 'seq_label_pairs_enc' if self.data_name == 'ag_news' else 'seq_tags_pairs_enc' # Key in dataset - semantically correct for the task at hand.

        # Load pre-processed data
        path_data = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'data.json')
        path_vocab = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'vocabs.json')
        data = load_json(path_data)
        self.vocab = load_json(path_vocab)       # Required for decoding sequences for interpretations. TODO: Find suitable location... or leave be...
        self.vocab_size = len(self.vocab['words'])  # word vocab is used for model dimensionality setting

        self.datasets = dict()
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
                self.test_dataloader = DataLoader(dataset=split_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            if split == 'valid':
                self.val_dataloader = DataLoader(dataset=split_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        print('---- REAL DATA SUCCESSFULLY INITIALISED ----')

    def _init_models(self):
        """ Initialises models, loss functions, optimisers and sets models to training mode """

        # Models
        # TODO: fix implementation to be consistent between models for parameter passing
        self.task_learner = TaskLearner(**self.model_config['TaskLearner']['Parameters'], vocab_size=self.vocab_size, tagset_size=self.tag_space_size, task_type=self.task_type).to(self.device)
        self.svae = SVAE(config=self.config, vocab_size=self.vocab_size).to(self.device)
        self.discriminator = Discriminator(z_dim=self.model_config['Discriminator']['z_dim']).to(self.device)

        # Loss Functions
        # Note: svae loss function is not defined herein
        if self.task_type == 'NER':
            self.tl_loss_fn = nn.NLLLoss().to(self.device)
        if self.task_type == 'CLF':
            self.tl_loss_fn = nn.CrossEntropyLoss().to(self.device)

        self.dsc_loss_fn = nn.BCELoss().to(self.device)

        # Optimisers
        self.tl_optim = optim.SGD(self.task_learner.parameters(), lr=self.learning_rates['task_learner'])
        self.svae_optim = optim.Adam(self.svae.parameters(), lr=self.learning_rates['svae'])
        self.dsc_optim = optim.Adam(self.discriminator.parameters(), lr=self.learning_rates['discriminator'])

        # Training Modes
        self.task_learner.train()
        self.svae.train()
        self.discriminator.train()

        print('---- MODELS SUCCESSFULLY INITIALISED ----')

    def train(self):
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
        """
        # Get dataset... for now is training
        self.train_dataset_l, self.train_dataset_u = split_data(dataset=self.datasets['train'], splits=(0.1,0.9))
        self.train_dataloader_l = DataLoader(dataset=self.train_dataset_l, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.train_dataloader_u = DataLoader(dataset=self.train_dataset_u, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        best_performance = 0
        step = 0    # Used for KL annealing
        for epoch in range(self.epochs):
            
            batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(self.train_dataloader_l))
            batch_sequences_u, batch_lengths_u, _ = next(iter(self.train_dataloader_u))

            if torch.cuda.is_available():
                batch_sequences_l = batch_sequences_l.to(self.device)
                batch_lengths_l = batch_lengths_l.to(self.device)
                batch_tags_l = batch_tags_l.to(self.device)
                batch_sequences_u = batch_sequences_u.to(self.device)
                batch_length_u = batch_lengths_u.to(self.device)
            
            # Strip off tag padding and flatten
            # this occurs to the sequences of tokens in the forward pass of the RNNs
            # we do it here to match them and make loss computations faster
            batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                            batch_sequences=batch_tags_l,
                                            pad_idx=self.pad_idx).view(-1)

            # Task Learner Step
            self.tl_optim.zero_grad()   # TODO: confirm if this gradient zeroing is correct
            tl_preds = self.task_learner(batch_sequences_l, batch_lengths_l)
            tl_loss = self.tl_loss_fn(tl_preds, batch_tags_l)
            tl_loss.backward()
            self.tl_optim.step()

            self.tb_writer.add_scalar('Loss/TaskLearner/train', tl_loss, epoch)

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
                                                                step=step,      # TODO: review how steps work when nested looping on each epoch.. I assume it's the same as SVAE step == epoch
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
                                                                step=step,      # TODO: review how steps work when nested looping on each epoch.. I assume it's the same as SVAE step == epoch
                                                                k=self.model_config['SVAE']['k'],
                                                                x0=self.model_config['SVAE']['x0'])

                self.tb_writer.add_scalar('Utils/SVAE/train/kl_weight_l', KL_weight_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Utils/SVAE/train/kl_weight_u', KL_weight_u, i + (epoch*self.svae_iterations))

                svae_loss_l = (NLL_loss_l + KL_weight_l * KL_loss_l) / batch_size_l
                svae_loss_u = (NLL_loss_u + KL_weight_u * KL_loss_u) / batch_size_u

                # Adversary loss - trying to fool the discriminator!
                dsc_preds_l = self.discriminator(mean_l)
                dsc_preds_u = self.discriminator(mean_u)

                dsc_real_l = torch.ones(batch_size_l)
                dsc_real_u = torch.ones(batch_size_u)

                if torch.cuda.is_available():
                    dsc_real_l = dsc_real_l.to(self.device)
                    dsc_real_u = dsc_real_u.to(self.device)

                adv_dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)
                adv_dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)
                adv_dsc_loss = adv_dsc_loss_l + adv_dsc_loss_u

                # Add scalar for adversarial loss
                self.tb_writer.add_scalar('Loss/SVAE/train/labelled/ADV', NLL_loss_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/unabelled/ADV', NLL_loss_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/ADV_total', NLL_loss_l, i + (epoch*self.svae_iterations))


                total_svae_loss = svae_loss_u + svae_loss_l + self.adv_hyperparam * adv_dsc_loss        # TODO: Review adversarial hyperparameter for SVAE loss func
                self.svae_optim.zero_grad()
                total_svae_loss.backward()
                self.svae_optim.step()

                # Add scalars for ELBO, NLL, KL divergence, and Total loss 
                self.tb_writer.add_scalar('Loss/SVAE/train/labelled/NLL', NLL_loss_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled/NLL', NLL_loss_u, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/labelled/KL_loss', KL_loss_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled/KL_loss', KL_loss_u, i + (epoch*self.svae_iterations))

                self.tb_writer.add_scalar('Loss/SVAE/train/labelled/total', svae_loss_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled/total', svae_loss_u, i + (epoch*self.svae_iterations))


                # Sample new batch of data while training adversarial network
                if i < self.svae_iterations - 1:
                    # TODO: strip out unnecessary information - investigate why output labels are required for SVAE loss function...
                    batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(self.train_dataloader_l))
                    batch_sequences_u, batch_length_u, _ = next(iter(self.train_dataloader_u))

                    if torch.cuda.is_available():
                        batch_sequences_l = batch_sequences_l.to(self.device)
                        batch_lengths_l = batch_lengths_l.to(self.device)
                        batch_tags_l = batch_tags_l.to(self.device)
                        batch_sequences_u = batch_sequences_u.to(self.device)
                        batch_length_u = batch_length_u.to(self.device)
                    
                    # Strip off tag padding and flatten
                    batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                                    batch_sequences=batch_tags_l,
                                                    pad_idx=self.pad_idx).view(-1)

            # SVAE Epoch loss after iterative cycle
            self.tb_writer.add_scalar('Loss/SVAE/train/Total', total_svae_loss, epoch)



            # Discriminator Step
            # TODO: Confirm that correct input is flowing into discriminator forward pass
            for j in range(self.dsc_iterations):

                with torch.no_grad():
                    _, mean_l, _, _ = self.svae(batch_sequences_l, batch_lengths_l)
                    _, mean_u, _, _ = self.svae(batch_sequences_u, batch_lengths_u)

                dsc_preds_l = self.discriminator(mean_l)
                dsc_preds_u = self.discriminator(mean_u)

                dsc_real_l = torch.ones(batch_size_l)
                dsc_real_u = torch.zeros(batch_size_u)

                if torch.cuda.is_available():
                    dsc_real_l = dsc_real_l.to(self.device)
                    dsc_real_u = dsc_real_u.to(self.device)

                dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)
                dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)

                total_dsc_loss = dsc_loss_l + dsc_loss_u
                self.dsc_optim.zero_grad()
                total_dsc_loss.backward()
                self.dsc_optim.step()

                # Sample new batch of data while training adversarial network
                # TODO: investigate why we need to do this, likely as the task learner is stronger than the adversarial/svae?
                if j < self.dsc_iterations - 1:
                    # TODO: strip out unnecessary information
                    batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(self.train_dataloader_l))
                    batch_sequences_u, batch_length_u, _ = next(iter(self.train_dataloader_u))

                    if torch.cuda.is_available():
                        batch_sequences_l = batch_sequences_l.to(self.device)
                        batch_lengths_l = batch_lengths_l.to(self.device)
                        batch_tags_l = batch_tags_l.to(self.device)
                        batch_sequences_u = batch_sequences_u.to(self.device)
                        batch_length_u = batch_length_u.to(self.device)
                    
                    # Strip off tag padding and flatten
                    batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                                    batch_sequences=batch_tags_l,
                                                    pad_idx=self.pad_idx).view(-1)

            self.tb_writer.add_scalar('Loss/Discriminator/train/labelled', dsc_loss_l, i + (epoch*self.dsc_iterations))
            self.tb_writer.add_scalar('Loss/Discriminator/train/unlabelled', dsc_loss_u, i + (epoch*self.dsc_iterations))

            self.tb_writer.add_scalar('Loss/Discriminator/train', total_dsc_loss, i + (epoch*self.dsc_iterations))
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch} - Losses (TL-{self.task_type} {tl_loss:0.2f} | SVAE {total_svae_loss:0.2f} | Disc {total_dsc_loss:0.2f})')
            
            if epoch % 100 == 0:
                # Check accuracy/F1 of task learner on validation set
                val_metrics = self.evaluation(task_learner=self.task_learner,
                                                        dataloader=self.val_dataloader,
                                                        task_type=self.task_type)
                
                # Returns tuple if NER otherwise singular variable if CLF
                val_string = f'F1 Scores - Macro {val_metrics[0]*100:0.2f}% Micro {val_metrics[1]*100:0.2f}%' if self.task_type == 'NER' else f'Accuracy {val_metrics*100:0.2f}'
                print(f'Task Learner ({self.task_type}) Validation {val_string}')

                if self.task_type == 'NER':
                    self.tb_writer.add_scalar('Metrics/TaskLearner/val/f1_macro', val_metrics[0]*100, epoch)
                    self.tb_writer.add_scalar('Metrics/TaskLearner/val/f1_micro', val_metrics[1]*100, epoch)
                if self.task_type == 'CLF':
                    self.tb_writer.add_scalar('Metrics/TaskLearner/val/acc', val_metrics, epoch)

                # best_performance - TODO: implement this

            step += 1

        # Compute final performance
        val_metrics = self.evaluation(task_learner=self.task_learner,
                                        dataloader=self.test_dataloader,
                                        task_type=self.task_type)

        val_string = f'F1 Scores - Macro {val_metrics[0]*100:0.2f}% Micro {val_metrics[1]*100:0.2f}%' if self.task_type == 'NER' else f'Accuracy {val_metrics*100:0.2f}'        
        print(f'Task Learner ({self.task_type}) Test {val_string}')



    def evaluation(self, task_learner, dataloader, task_type):
        """ Computes performance metrics on holdout sets (val, train)
        
        Arguments
        ---------
            task_learner : TODO
                TODO
            dataloader : TODO
                TODO
            task_type : str
                Type of model task e.g. CLF or NER
        
        Returns
        -------
            metric : float
                Accuracy (CLF) or F1 score (NER)
        
        Notes
        -----

        """

        task_learner.eval() # allows evaluations to be made, need to reset afterwares though.
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

        if task_type == 'NER':
            f1_macro = f1_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='macro')
            f1_micro = f1_score(y_true=true_labels_all.cpu().numpy(), y_pred=preds_all.cpu().numpy(), average='micro')
            return (f1_macro, f1_micro)
        if task_type == 'CLF':
            # Returns accuracy and a placeholder variable that can be ignored
            acc = accuracy_score(y_true=true_labels_all.cpu().nump(), y_pred=preds_all.cpu().numpy())
            return acc





class Tests(unittest.TestCase):
    def setUp(self):
        # Note: Tests are done with generated data incase real data isn't avaiable
        pass

    def test_data_init(self):
        pass

    def test_models_init(self):
        pass

    def test_train(self):
        # Test for both classification and seuqence models

        # test CLF

        # test SEQ

        pass


def main(config):
    # Train S-VAAL model
    Trainer(config)

    unittest.main()


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