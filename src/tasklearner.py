"""
Contains model initialisation procedures and test functionality for task learner.
There are two task learner configurations: 1. Text classifiation and 2. Sequence tagging

@author: Tyler Bikaun
"""

# Imports
import yaml
import numpy as np
import unittest

from data_generator import DataGenerator
from utils import to_var, trim_padded_seqs

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
Tensor = torch.Tensor

class TaskLearnerClassification(nn.Module):
    """ Initialises a many-to-one text classification task learner 
    
    Initial model architecture from: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

    Inital dataset: AG_NEWS (4 class - Word, Sports, Business, Sci/Tec)

    Arguments
    ---------
        TODO : TODO
            TODO
    
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TaskLearnerClassification, self).__init_()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, spase=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class TaskLearnerSequence(nn.Module):
    """ Initialises a sequence based task learner (RNN based) 
    
    Arguments
    ---------
        embedding_dim : int

        hidden_dim : int

        vocab_size : int

        tagset_size : int
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super(TaskLearnerSequence, self).__init__()

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # TODO: Implement padding_idx=self.pad_idx

        # Current sequence tagger is an LSTM (TODO: implement more advanced sequence taggers and options)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer that maps hidden state space from LSTM to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, batch_sequences: Tensor, batch_lengths: Tensor) -> Tensor:
        """
        Forward pass through Task Learner

        Arguments
        ---------
            batch_sequences : Tensor
                Batch of sequences
            batch_lengths : Tensor
                Batch of sequence lengths

        Returns
        -------
            tag_scores : Tensor
                Batch of predicted tag scores
        """
        # Sort and pack padded sequence for variable length LSTM
        sorted_lengths, sorted_idx = torch.sort(batch_lengths, descending=True)
        batch_sequences = batch_sequences[sorted_idx]
        input_embeddings = self.word_embeddings(batch_sequences)
        packed_input = rnn_utils.pack_padded_sequence(input_embeddings, sorted_lengths.data.tolist(), batch_first=True)
        
        lstm_out, _ = self.lstm(packed_input)
        
        # Unpack padded sequence
        padded_outputs = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        
        # Project into tag space
        tag_space = self.hidden2tag(padded_outputs.view(-1, padded_outputs.size(2)))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class Tester(DataGenerator):
    """ Tests task learner components
    Arguments
    ---------
        config : yaml
            Configuration file for model initialisation and testing 
    """
    def __init__(self, config):
        DataGenerator.__init__(self, config)   # Allows access properties and build methods
        self.config = config
        
        # Testing data properties
        self.batch_size = config['Tester']['batch_size']
        self.max_sequence_length = config['Tester']['max_sequence_length']
        self.embedding_dim = 128
        
        # Test run properties
        self.model_type = config['Tester']['model_type'].lower()
        self.epochs = config['Tester']['epochs']

        # Exe
        self.training_routine()
        
    def init_data(self):
        """ Initialise synthetic sequence data for testing """
        if self.model_type in ['task_learner', 'svae']:
            sequences, lengths = self.build_sequences(no_sequences=self.batch_size, max_sequence_length=self.max_sequence_length)
            self.dataset = self.build_sequence_tags(sequences=sequences, lengths=lengths)
            self.vocab = self.build_vocab(sequences)
            self.vocab_size = len(self.vocab)
        elif self.model_type == 'discriminator':
            self.dataset = self.build_latents(batch_size=self.batch_size, z_dim=self.z_dim)
    
    def init_model(self):
        """ Initialise neural network components including loss functions, optimisers and auxilliary functions """

        self.model = TaskLearner(embedding_dim=self.embedding_dim, hidden_dim=128, vocab_size=self.vocab_size, tagset_size=self.tag_space_size).cuda()
        self.loss_fn = nn.NLLLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=0.1)
        # Set model to train mode
        self.model.train()

    def training_routine(self):
        """ Abstract training routine """
        # Initialise training data and model for testing
        print(f'TRAINING {self.model_type.upper()}')
        self.init_data()
        self.init_model()

        # Train model
        for epoch in range(self.epochs):
            for batch_sequences, batch_lengths, batch_tags in self.dataset:

                if torch.cuda.is_available():
                    batch_sequences = batch_sequences.cuda()
                    batch_lengths = batch_lengths.cuda()
                    batch_tags = batch_tags.cuda()

                if epoch == 0:
                    print(f'Shapes | Sequences: {batch_sequences.shape} Lengths: {batch_lengths.shape} Tags: {batch_tags.shape}')

                batch_size = batch_sequences.size(0)

                self.model.zero_grad()
                # Strip off tag padding (similar to variable length sequences via pack padded methods)
                batch_tags = trim_padded_seqs(batch_lengths=batch_lengths,
                                            batch_sequences=batch_tags,
                                            pad_idx=self.pad_idx).view(-1)

                # Forward pass through model
                tag_scores = self.model(batch_sequences, batch_lengths)
                
                # Calculate loss and backpropagate error through model
                loss = self.loss_fn(tag_scores, batch_tags)
                loss.backward()
                self.optim.step()
                
                print(f'Epoch: {epoch} - Loss: {loss.data.detach():0.2f}')
                step += 1


class Tests(unittest.TestCase):
    def setUp(self):
        # Init class
        self.sampler = Sampler(config='x', budget=10, sample_size=2)
        # Init random tensor
        self.data = torch.rand(size=(10,2,2))  # dim (batch, length, features)

    def test_sample_random(self):
        self.assertEqual(self.sampler.sample_random(self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_random(self.data).shape[0], self.sampler.sample_size)

def main(config):
    """
    Initialises models, generates synthetic sequence data and runs tests on each model
    to ensure they're working correctly.
    """
    # Generate synthetic data
    # data_generator = DataGenerator(config)
    # sequences, lengths = data_generator.build_sequences(batch_size=2, max_sequence_length=10)
    # test_dataset = data_generator.build_sequence_tags(sequences=sequences, lengths=lengths)
    # vocab = data_generator.build_vocab(sequences)

    # Initialise models
    # pass

    # Run tests
    Tester(config)

    # Run tests
    # unittest.main()



if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)