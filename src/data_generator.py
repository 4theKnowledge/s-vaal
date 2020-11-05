"""
Data generator is a module for generating situational data for testing components of S-VAAL.

@author: Tyler Bikaun
"""

# Imports
import numpy as np
import random
import yaml
import math

import torch
from torch.utils.data import Dataset, DataLoader
Tensor = torch.Tensor


class DataGenerator:
    def __init__(self, config):
        self.pad_idx = config['Utils']['special_tokens']['pad_idx']
        self.special_chars_list = [self.pad_idx]
        self.no_output_classes = len(config['Model']['output_classes'])
        self.tag_space_size = self.no_output_classes + len(self.special_chars_list)
        
        print(f'Using label space size of: {self.tag_space_size}')
        
    def build_sequences(self, no_sequences: int, max_sequence_length: int) -> Tensor:
        """
        Builds tensor of specified size containing variable length, padded, sequences of integers
            
        Arguments
        ---------
            no_sequences : int
                Number of sequences to generate
            max_sequence_length : int
                Maximum length of sequences
        Returns
        -------
            sequences : Tensor
                Batch of generated sequences
            lengths : Tensor
                Batch of generated sequence lengths
        """
        seqs = list()
        for _ in range(no_sequences):
            # Generate random integer sequences
            # sequence must be at least 1 token long...
            # range of token ints are low=1 (0 is for padding) to high=no_sequences * max_seq_length + 1 (if ever word was unique) 
            seq = np.random.randint(low=1, high=no_sequences*max_sequence_length, size=(random.randint(1, max_sequence_length),))
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
            sequences : Tensor
                Batch of sequences
            lengths : Tensor
                Batch of sequence lengths
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

    def build_vocab(self, sequences: Tensor) -> list:
        """ Builds vocabulary from sequence data 
        
        
        Note: due to the way data is generated, the vocabulary needs to be at least as big as the largest integer, even
        if there isn't that many unique tokens in the sequences. Therefore, we will artificially create the vocab here.

        Arguments
        ---------
            sequences : Tensor
                Batch of sequences
        Returns
        -------
            vocab : list, int
                List of integers correponding to vocabulary of word-index mappings

        """
        # CREATING REAL VOCAB FROM SEQUENCES
        # vocab = list()
        # for sequence in sequences:
        #     vocab.extend(sequence.tolist())
        # vocab = list(set(vocab))
        # print(f'Generated vocabulary with {len(vocab)} terms ')
        # return vocab
        
        vocab = range(1, max(sequences.view(-1).tolist())+2,1)
        print(f'Generated vocabulary with {len(vocab)} terms (min {min(vocab)} max {max(vocab)})')

        return vocab

    def build_latents(self, no_sequences: int, z_dim: int) -> Tensor:
        """ Generates tensor of randomised latent data sampled from a standard normal distribution 
        
        Arguments
        ---------
            no_sequences : int
                Number of latent space features to generate
            z_dim : int
                Dimension of latent space
        Returns
        -------
            latents : Tensor
                Batch of generated latent space features
        
        """
        latents = torch.randn(size=(no_sequences,z_dim))
        print(f'Generated latent data with shape {latents.shape}')
        return latents

    def build_datasets(self, no_sequences: int, max_sequence_length: int, split: float) -> Tensor:
        """ Builds labelled and unlabelled datasets for active learning
        
        Arguments
        ---------
            no_sequences : int
                Number of sequences to generate
            max_sequence_length : int
                Maximum length of sequences
        Returns
        -------
            dataset_l : Tensor
                Labelled dataset
            dataset_u : Tensor
                Unlabelled dataset
        
        """
        print('Generating labelled/unlabelled datasets')
        
        assert split <= 0.25

        no_samples_l = math.ceil(no_sequences*split)
        no_samples_u = no_sequences - no_samples_l
        # print(f'Sample numbers: Labelled {no_samples_l} - Unlabelled {no_samples_u}')

        seqs_l, lens_l = self.build_sequences(no_sequences=no_samples_l, max_sequence_length=max_sequence_length)
        seqs_u, lens_u = self.build_sequences(no_sequences=no_samples_u, max_sequence_length=max_sequence_length)

        dataset_l = self.build_sequence_tags(seqs_l, lens_l)
        dataset_u = self.build_sequence_tags(seqs_u, lens_u)
        
        vocab = self.build_vocab(seqs_u)

        return dataset_l, dataset_u, vocab


class SequenceDataset(Dataset, DataGenerator):
    """ Sequence dataset """

    def __init__(self, config, no_sequences, max_sequence_length):
        DataGenerator.__init__(self, config)
        sequences, sequence_lengths = self.build_sequences(no_sequences=no_sequences, max_sequence_length=max_sequence_length)
        self.sequences, self.sequence_lengths, self.sequence_tags = self.build_sequence_tags(sequences, sequence_lengths)[0]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sequences[idx], self.sequence_lengths[idx], self.sequence_tags[idx]


def main(config):
    """ Runs basic functions for consistency and functionality checking """
    # tester = DataGenerator(config)

    # Generate sequences and their corresponding lengths
    # sequences, lengths = tester.build_sequences(no_sequences=2, max_sequence_length=10)
    
    # Generate output tags and build dataset with generated sequences, lengths and tags
    # tester.build_sequence_tags(sequences=sequences, lengths=lengths)
    
    # Generate vocabulary from sequences
    # tester.build_vocab(sequences)
    
    # Generate latents
    # tester.build_latents(no_sequences=2, z_dim=8)

    # Generate labelled/unlabelled datasets
    # dataset_l, dataset_u, vocab = tester.build_datasets(no_sequences=10, max_sequence_length=40, split=0.1)

    # Test dataset generator and dataloader
    sequence_dataset = SequenceDataset(config, no_sequences=100, max_sequence_length=30)
    dataloader = DataLoader(sequence_dataset, batch_size=7, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        X, lens, y = batch
        print(i, X.shape, lens.shape, y.shape)

if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)