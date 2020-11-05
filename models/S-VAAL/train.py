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
Tensor = torch.Tensor

from models import TaskLearner, SVAE, Discriminator
from utils import to_var, trim_padded_tags
from data_generator import DataGenerator


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
        self.svae_iterations = config['Train']['svae_steps']
        self.dsc_iterations = config['Train']['discriminator_steps']
        self.learning_rates = config['Train']['learning_rates']

        # Exe
        self.init_dataset()
        self.init_models()


    def init_dataset(self):
        """ Initialises dataset for model training """
        # Currently will be using generated data, but in the future will be real.
        sequences, lengths = self.build_sequences(batch_size=self.batch_size, max_sequence_length=self.max_sequence_length)
        self.dataset = self.build_sequence_tags(sequences=sequences, lengths=lengths)
        self.vocab = self.build_vocab(sequences)
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
        




        





def main(config):
    # Initiate S-VAAL training

    Trainer(config)


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)