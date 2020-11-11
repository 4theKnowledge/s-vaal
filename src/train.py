"""
Trainer for generalisation of S-VAAL model.

TODO:
- Add tensorboard logging
- Add model caching/saving
- Add model restart/checkpointing

@author: Tyler Bikaun
"""

# Imports
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
Tensor = torch.Tensor

from tasklearner import TaskLearnerSequence as TaskLearner  # Change import alias if using both models.
from models import SVAE, Discriminator
from utils import to_var, trim_padded_seqs
from data_generator import DataGenerator, SequenceDataset


class Trainer(DataGenerator):
    """ Prepares and trains S-VAAL model """
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        DataGenerator.__init__(self, config)

        self.config = config
        self.model_config = config['Model']

        self.tb_writer = SummaryWriter()

        # Testing data properties
        self.batch_size = config['Tester']['batch_size']
        self.max_sequence_length = config['Tester']['max_sequence_length']
        
        # Test run properties
        self.epochs = config['Train']['epochs']
        self.svae_iterations = config['Train']['svae_iterations']
        self.dsc_iterations = config['Train']['discriminator_iterations']
        self.learning_rates = config['Train']['learning_rates']
        self.adv_hyperparam = config['Train']['adversarial_hyperparameter']

        # Exe
        self.init_dataset()
        self.init_models()
        self.train()

    def init_dataset(self):
        """ Initialises dataset for model training """
        # Currently will be using generated data, but in the future will be real.

        self.dataset_l = SequenceDataset(config, no_sequences=8, max_sequence_length=self.max_sequence_length)
        self.dataloader_l = DataLoader(self.dataset_l, batch_size=2, shuffle=True, num_workers=0)

        self.dataset_u = SequenceDataset(config, no_sequences=16, max_sequence_length=self.max_sequence_length)
        self.dataloader_u = DataLoader(self.dataset_u, batch_size=2, shuffle=True, num_workers=0)

        # Concatenate sequences in X_l and X_u to build vocabulary for downstream
        self.vocab = self.build_vocab(sequences = torch.cat((self.dataset_l.sequences, self.dataset_u.sequences)))
        self.vocab_size = len(self.vocab)

        print('---- DATA SUCCESSFULLY INITIALISED ----')

    def init_dataset_real(self):
        """
        Initialise real datasets by reading encoding data
        """
        pass

        


    def init_models(self):
        """ Initialises models, loss functions, optimisers and sets models to training mode """

        # Models
        # TODO: fix implementation to be consistent between models for parameter passing
        self.task_learner = TaskLearner(**self.model_config['TaskLearner']['Parameters'], vocab_size=self.vocab_size, tagset_size=self.tag_space_size).cuda()
        self.svae = SVAE(config=self.config, vocab_size=self.vocab_size).cuda()
        self.discriminator = Discriminator(z_dim=self.model_config['Discriminator']['z_dim']).cuda()

        # Loss Functions
        # Note: svae loss function is not defined herein
        self.tl_loss_fn = nn.NLLLoss()
        self.dsc_loss_fn = nn.BCELoss()

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
        step = 0    # Used for KL annealing

        for epoch in range(self.epochs):

            batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(self.dataloader_l))
            batch_sequences_u, batch_lengths_u, _ = next(iter(self.dataloader_u))


            if torch.cuda.is_available():
                batch_sequences_l = batch_sequences_l.cuda()
                batch_lengths_l = batch_lengths_l.cuda()
                batch_tags_l = batch_tags_l.cuda()
                batch_sequences_u = batch_sequences_u.cuda()
                batch_length_u = batch_lengths_u.cuda()
            
            # Strip off tag padding and flatten
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

                svae_loss_l = (NLL_loss_l + KL_weight_l * KL_loss_l) / batch_size_l
                svae_loss_u = (NLL_loss_u + KL_weight_u * KL_loss_u) / batch_size_u

                # Adversary loss - trying to fool the discriminator!
                dsc_preds_l = self.discriminator(mean_l)
                dsc_preds_u = self.discriminator(mean_u)

                dsc_real_l = torch.ones(batch_size_l)
                dsc_real_u = torch.ones(batch_size_u)

                if torch.cuda.is_available():
                    dsc_real_l = dsc_real_l.cuda()
                    dsc_real_u = dsc_real_u.cuda()

                adv_dsc_loss = self.dsc_loss_fn(dsc_preds_l, dsc_real_l) + self.dsc_loss_fn(dsc_preds_u, dsc_real_u)

                total_svae_loss = svae_loss_u + svae_loss_l + self.adv_hyperparam * adv_dsc_loss
                self.svae_optim.zero_grad()
                total_svae_loss.backward()
                self.svae_optim.step()

                self.tb_writer.add_scalar('Loss/SVAE/train/labelled', svae_loss_l, i + (epoch*self.svae_iterations))
                self.tb_writer.add_scalar('Loss/SVAE/train/unlabelled', svae_loss_u, i + (epoch*self.svae_iterations))


                # Sample new batch of data while training adversarial network
                if i < self.svae_iterations - 1:
                    # TODO: strip out unnecessary information - investigate why output labels are required for SVAE loss function...
                    batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(self.dataloader_l))
                    batch_sequences_u, batch_length_u, _ = next(iter(self.dataloader_u))

                    if torch.cuda.is_available():
                        batch_sequences_l = batch_sequences_l.cuda()
                        batch_lengths_l = batch_lengths_l.cuda()
                        batch_tags_l = batch_tags_l.cuda()
                        batch_sequences_u = batch_sequences_u.cuda()
                        batch_length_u = batch_length_u.cuda()
                    
                    # Strip off tag padding and flatten
                    batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                                    batch_sequences=batch_tags_l,
                                                    pad_idx=self.pad_idx).view(-1)

            self.tb_writer.add_scalar('Loss/SVAE/train', total_svae_loss, i + (epoch*self.svae_iterations))

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
                    dsc_real_l = dsc_real_l.cuda()
                    dsc_real_u = dsc_real_u.cuda()

                total_dsc_loss = self.dsc_loss_fn(dsc_preds_l, dsc_real_l) + self.dsc_loss_fn(dsc_preds_u, dsc_real_u)
                self.dsc_optim.zero_grad()
                total_dsc_loss.backward()
                self.dsc_optim.step()

                # Sample new batch of data while training adversarial network
                # TODO: investigate why we need to do this, likely as the task learner is stronger than the adversarial/svae?
                if j < self.dsc_iterations - 1:
                    # TODO: strip out unnecessary information
                    batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(self.dataloader_l))
                    batch_sequences_u, batch_length_u, _ = next(iter(self.dataloader_u))

                    if torch.cuda.is_available():
                        batch_sequences_l = batch_sequences_l.cuda()
                        batch_lengths_l = batch_lengths_l.cuda()
                        batch_tags_l = batch_tags_l.cuda()
                        batch_sequences_u = batch_sequences_u.cuda()
                        batch_length_u = batch_length_u.cuda()
                    
                    # Strip off tag padding and flatten
                    batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                                    batch_sequences=batch_tags_l,
                                                    pad_idx=self.pad_idx).view(-1)
    
            self.tb_writer.add_scalar('Loss/Discriminator/train', total_dsc_loss, i + (epoch*self.dsc_iterations))


            print(f'Epoch {epoch} - Losses (TL {tl_loss:0.2f} | SVAE {total_svae_loss:0.2f} | Disc {total_dsc_loss:0.2f})')
            step += 1

def main(config):
    # Train S-VAAL model
    Trainer(config)


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    # Seeds
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed']

    main(config)