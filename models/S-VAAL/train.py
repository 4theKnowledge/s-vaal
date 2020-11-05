"""
Trainer for generalisation of S-VAAL model.

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
Tensor = torch.Tensor

from models import TaskLearner, SVAE, Discriminator
from utils import to_var, trim_padded_tags
from data_generator import DataGenerator, SequenceDataset


class Trainer(DataGenerator):
    """ Prepares and trains S-VAAL model """
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        DataGenerator.__init__(self, config)

        self.config = config
        self.model_config = config['Model']

        # Testing data properties
        self.batch_size = config['Tester']['batch_size']
        self.max_sequence_length = config['Tester']['max_sequence_length']
        
        # Test run properties
        self.epochs = config['Train']['epochs']
        self.svae_iterations = config['Train']['svae_iterations']
        self.dsc_iterations = config['Train']['discriminator_iterations']
        self.learning_rates = config['Train']['learning_rates']

        # Exe
        self.init_dataset()
        self.init_models()
        self.train()

    def init_dataset(self):
        """ Initialises dataset for model training """
        # Currently will be using generated data, but in the future will be real.

        self.dataset_l = SequenceDataset(config, no_sequences=8, max_sequence_length=30)
        self.dataloader_l = DataLoader(self.dataset_l, batch_size=2, shuffle=True, num_workers=0)

        self.dataset_u = SequenceDataset(config, no_sequences=16, max_sequence_length=30)
        self.dataloader_u = DataLoader(self.dataset_u, batch_size=2, shuffle=True, num_workers=0)

        # Concatenate sequences in X_l and X_u to build vocabulary for downstream
        self.vocab = self.build_vocab(sequences = torch.cat((self.dataset_l.sequences, self.dataset_u.sequences)))
        self.vocab_size = len(self.vocab)

        print('---- DATA SUCCESSFULLY INITIALISED ----')

    def init_models(self):
        """ Initialises models, loss functions, optimisers and sets models to training mode """

        # Models
        # TODO: fix implementation to be consistent between models for parameter passing
        self.task_learner = TaskLearner(**self.model_config['TaskLearner']['Parameters'], vocab_size=self.vocab_size, tagset_size=self.tag_space_size)
        self.svae = SVAE(config=self.config, vocab_size=self.vocab_size)
        self.discriminator = Discriminator(z_dim=self.model_config['Discriminator']['z_dim'])

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
        Sequentially trains S-VAAL

        Training sequence
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


        Returns
        -------


        """
        step = 0    # Used for KL annealing

        for epoch in range(self.epochs):

            

            # batch_sequences_l, batch_lengths_l, batch_tags_l = next(dataset_l)  # some sort of generator around labelled dataset
            # batch_sequences_u, batch_lengths_u, _ = next(dataset_u) # some sort of generator, generated data has labelled but wont use here
            
            # for batch_sequences, batch_lengths, batch_tags in self.dataset:

            #     # Strip off tag padding and flatten
            #     batch_tags = trim_padded_tags(batch_lengths=batch_lengths,
            #                                     batch_tags=batch_tags,
            #                                     pad_idx=self.pad_idx).view(-1)

            #     # Task Learner Step
            #     self.tl_optim.zero_grad()   # TODO: confirm if this gradient zeroing is correct
            #     tl_preds = self.task_learner(batch_sequences, batch_lengths)
            #     tl_loss = self.tl_loss_fn(tl_preds, batch_tags)
            #     tl_loss.backward()
            #     self.tl_optim.step()

            #     # Used to normalise SVAE loss
            #     batch_size = batch_sequences.size(0)
            #     # SVAE Step
            #     # TODO: Extend for unsupervised and supervised losses
            #     #       - As well as add in discriminator losses etc.
            #     for i in range(self.svae_iterations):
            #         self.svae_optim.zero_grad()
            #         logp, mean, logv, z = self.svae(batch_sequences, batch_lengths)
            #         NLL_loss, KL_loss, KL_weight = self.svae.loss_fn(
            #                                                         logp=logp,
            #                                                         target=batch_tags,
            #                                                         length=batch_lengths,
            #                                                         mean=mean,
            #                                                         logv=logv,
            #                                                         anneal_fn=self.model_config['SVAE']['anneal_function'],
            #                                                         step=step,      # TODO: review how steps work when nested looping on each epoch.. I assume it's the same as SVAE step == epoch
            #                                                         k=self.model_config['SVAE']['k'],
            #                                                         x0=self.model_config['SVAE']['x0'])
            #         svae_loss = (NLL_loss + KL_weight * KL_loss) / batch_size
            #         svae_loss.backward()
            #         self.svae_optim.step()

            #     # Discriminator Step
            #     for j in range(self.dsc_iterations):
            #         # self.dsc_optim.zero_grad()
            #         pass
                    




                
            
            # print(f'Epoch {epoch} - Losses (TL {tl_loss:0.2f} | SVAE {svae_loss:0.2f} | Disc {dsc_loss:0.2f})')
            # step += 1

def main(config):
    # Train S-VAAL model
    Trainer(config)


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)