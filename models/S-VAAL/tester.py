"""
Tester is a module for generating situational data for testing components of S-VAAL.

@author: Tyler Bikaun
"""

# Imports
import numpy as np
import random
import yaml

import torch
Tensor = torch.Tensor


class Tester:
    def __init__(self, config):
        self.pad_idx = config['Utils']['special_tokens']['pad_idx']
        self.special_chars_list = [self.pad_idx]
        self.no_output_classes = len(config['Model']['output_classes'])
        self.tag_space_size = self.no_output_classes + len(self.special_chars_list)
        
        print(f'Using label space size of: {self.tag_space_size}')
        
    def build_sequences(self, batch_size: int, max_sequence_length: int) -> Tensor:
        """
        Builds tensor of specified size containing variable length, padded, sequences of integers
            
        Arguments
        ---------
            batch_size : int
                Number of sequences to generate
            max_seq_len : int
                Maximum length of sequences
        Returns
        -------
            sequences : tensor
                Tensor of generated sequences
            lengths : tensor
                Tensor of sequence lengths
        """
        seqs = list()
        for _ in range(batch_size):
            # Generate random integer sequences
            # sequence must be at least 1 token long...
            seq = np.random.randint(low=1, high=100, size=(random.randint(1, max_sequence_length),))
            # Add padding
            seq = np.concatenate((seq, np.ones(shape=(max_sequence_length - len(seq)))*self.pad_idx), axis=None)
            seqs.append(seq)
        sequences = torch.LongTensor(seqs)
        lengths = torch.tensor([len(seq[seq != self.pad_idx]) for seq in sequences])
        
        print(f'Generated sequences with the following shapes - Sequences: {sequences.shape}\tLengths: {lengths.shape}')
        
        return sequences, lengths
    
    def build_sequence_tags(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        """
        Given a set of sequences, generates ground truth labels
        
        Labels need to be non-zero (otherwise get confused with special characters; currnetly only concerned about 0 = PAD)
        
        Arguments
        ---------
            sequences : tensor
                Tensor of generated sequences
            label_space_size : int
                Size of tag space
        Returns
        -------
            X, lengths, y : list of tuples
                Artificial ground truth dataset
                    X dim : (seq len, batch size )
                    lengths dim : (batch size)
                    y dim : (batch size, 1)
        """
        
        dataset = list()    # stores batch of data (X, lens, y)
        
        global_tag_list = list()
        
        for sequence in sequences:
            # Each 'token' in the sequence has a tag mapping
            tag_list = list()
            for token in sequence:
                if token != self.pad_idx:   # don't give a label to any padding...
                    tag_list.append(random.randint(1, self.tag_space_size-1))   # need to minus 1 as output loss function indexes from 0 to n_class - 1
                else:
                    tag_list.append(self.pad_idx)
            
            global_tag_list.append(torch.LongTensor(tag_list))
        
        global_tag_tensor = torch.stack(global_tag_list)
        dataset.append((sequences, lengths, global_tag_tensor))   # stack list of labels into tensors

        print(f'Generated dataset with the following shapes - Sequences: {sequences.shape}\tLengths: {lengths.shape}\tTags: {global_tag_tensor.shape}')

        return dataset


def main(config):
    # print(config['Utils']['special_tokens'])
    tester = Tester(config)

    # Generate sequences and their corresponding lengths
    sequences, lengths = tester.build_sequences(batch_size=2, max_sequence_length=10)
    # Generate output tags and build dataset with generated sequences, lengths and tags
    dataset = tester.build_sequence_tags(sequences=sequences, lengths=lengths)

if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)