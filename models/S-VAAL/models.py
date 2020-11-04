"""
Contains model initialisation information and procedures

@author: Tyler Bikaun
"""

# Imports
import yaml

from data_generator import DataGenerator

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
Tensor = torch.Tensor



class TaskLearner:
    """ Sequence based task learner """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(TaskLearner, self).__init__()

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Current sequence tagger is an LSTM (TODO: implement more advanced sequence taggers and options)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer that maps hidden state space from LSTM to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)


    def forward(self, batch_sequences: Tensor, batch_lengths: Tensor) -> Tensor:
        """ """

        # Sort and pack padded sequence for variable length LSTM
        # batch_size = batch_sequences.size(0)
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
        # b, s, _ = padded_outputs.size()
        
        # Project into tag space
        tag_space = self.hidden2tag(padded_outputs.view(-1, padded_outputs.size(2)))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def tests(self, batch_sequences, batch_lengths, batch_tags):
        """ Runs checks of componentry for example forward pass and basic model training routine """
        
        



class SVAE:
    def __init__(self):
        pass

    def tests(self):
        """ Runs checks of componentry """
        pass

class Discriminator:
    def __init__(self):
        pass

    def tests(self):
        """ Runs checks of componentry """
        pass


def main(config):
    """
    Initialises models, generates synthetic sequence data and runs tests on each model
    to ensure they're working correctly.
    """
    # Generate synthetic data
    data_generator = DataGenerator(config)
    sequences, lengths = data_generator.build_sequences(batch_size=2, max_sequence_length=10)
    test_dataset = data_generator.build_sequence_tags(sequences=sequences, lengths=lengths)
    vocab = data_generator.build_vocab(sequences)

    # Initialise models
    pass


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)