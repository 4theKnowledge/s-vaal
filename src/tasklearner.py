"""
Contains model initialisation procedures and test functionality for task learner.
There are two task learner configurations: 1. Text classifiation (CLF) and 2. Sequence tagging (NER)

TODO:
    - Rename task type as CLF and SEQ rather than CLF and NER... makes more general for other SEQ tasks like POS
    - Add bidirectionality, GRU, RNN, multi-layers

@author: Tyler Bikaun
"""

# Imports
import yaml
import numpy as np
import unittest

from data import DataGenerator
from utils import to_var, trim_padded_seqs, split_data
from connections import load_config

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
Tensor = torch.Tensor


class TaskLearner(nn.Module):
    """ Initialises a task learner for either text classification or sequence labelling 
    
    Arguments
    ---------
        embedding_dim : int
            Size of embedding dimension.
        hidden_dim : int
            Size of hiddend dimension of rnn model
        vocab_size : int
            Size of input word vocabulary.
        tagset_size : int
            Size of output tag space. For CLF this will be 1, for NER this will be n.
        task_type : str
            Task type of the task learner e.g. CLF for text classification or NER for named entity recognition
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, task_type: str):
        super(TaskLearner, self).__init__()

        self.task_type = task_type  # CLF - text classification; NER - named entity recognition

        self.rnn_type = 'gru'

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # TODO: Implement padding_idx=self.pad_idx
        
        if self.rnn_type == 'gru':
            rnn = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn = nn.LSTM
        elif self.rnn_type == 'rnn':
            rnn = nn.RNN
        else:
            raise ValueError
        
        # Sequence tagger (TODO: implement more advanced sequence taggers and options)
        self.rnn = rnn(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        if self.task_type == 'NER':
            # Linear layer that maps hidden state space from rnn to tag space
            self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=tagset_size)

        if self.task_type == 'CLF':
            self.drop = nn.Dropout(p=0.5)
            self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=1)

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
        input_embeddings = self.word_embeddings(batch_sequences)

        # Sort and pack padded sequence for variable length LSTM
        sorted_lengths, sorted_idx = torch.sort(batch_lengths, descending=True)
        batch_sequences = batch_sequences[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input=input_embeddings,
                                                        lengths=sorted_lengths.data.tolist(),
                                                        batch_first=True)

        rnn_out, _ = self.rnn(packed_input)

        # Unpack padded sequence
        padded_outputs = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        if self.task_type == 'NER':
            # Project into tag space
            tag_space = self.hidden2tag(padded_outputs.view(-1, padded_outputs.size(2)))
            tag_scores = F.log_softmax(tag_space, dim=1)
            return tag_scores

        if self.task_type == 'CLF':
            output = self.drop(padded_outputs)
            output = self.hidden2tag(output)
            tag_scores = torch.sigmoid(output)
            return tag_scores
            

if __name__ == '__main__':
    # Seeds
    config = load_config()
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed'])