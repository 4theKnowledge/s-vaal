"""
Data is a module for working with datasets and generating situational data for testing components of S-VAAL.

TODO:
- Make sequences deterministic so that they can be used to validate simple feedforwards including latents

@author: Tyler Bikaun
"""

# Imports
import numpy as np
import random
import yaml
import math
import unittest

from utils import get_lengths
from connections import load_config

import torch
from torch.utils.data import Dataset, DataLoader
Tensor = torch.Tensor


class DataGenerator:
    def __init__(self):
        config = load_config()

        self.pad_idx = config['Utils']['special_token2idx']['<PAD>']
        self.special_chars_list = [self.pad_idx]
        output_classes = ['ORG', 'PER', 'LOC', 'MISC']
        self.no_output_classes = len(output_classes)
        self.tag_space_size = self.no_output_classes + len(self.special_chars_list)
        self.no_classes_clf = 4 # TODO: make more suitable...
        
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
        
        # print(f'Generated sequences with the following shapes - Sequences: {sequences.shape}\tLengths: {lengths.shape}')

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

        # print(f'Generated dataset with the following shapes - Sequences: {sequences.shape}\tLengths: {lengths.shape}\tTags: {global_tag_tensor.shape}')

        return dataset

    def build_sequence_classes(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        """ Builds sequence classes for classification tasks

        Arguments
        ---------

        Returns
        -------

        Notes
        -----

        """

        dataset = list()    # stores batch of data (X, lens, y)

        global_class_list = list()
        
        for _ in sequences:
            # Each sentence has a single class token
            global_class_list.append(torch.randint(low=0,high=self.no_classes_clf, size=(1,)))
        
        global_class_tensor = torch.stack(global_class_list)
        dataset.append((sequences, lengths, global_class_tensor))   # stack list of labels into tensors

        # print(f'Generated dataset with the following shapes - Sequences: {sequences.shape}\tLengths: {lengths.shape}\tTags: {global_class_tensor.shape}')

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
        # print(f'Generated vocabulary with {len(vocab)} terms (min {min(vocab)} max {max(vocab)})')

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
        # print(f'Generated latent data with shape {latents.shape}')
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
        # print('Generating labelled/unlabelled datasets')
        
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
    """ Generated dataset object for sequences """
    def __init__(self, no_sequences, max_sequence_length, task_type):
        DataGenerator.__init__(self)
        sequences, sequence_lengths = self.build_sequences(no_sequences=no_sequences, max_sequence_length=max_sequence_length)

        if task_type == 'NER':
            self.sequences, self.sequence_lengths, self.sequence_tags = self.build_sequence_tags(sequences, sequence_lengths)[0]
        elif task_type == 'CLF':
            self.sequences, self.sequence_lengths, self.sequence_tags = self.build_sequence_classes(sequences, sequence_lengths)[0]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sequences[idx], self.sequence_lengths[idx], self.sequence_tags[idx]


class RealDataset(Dataset):
    """ Real dataset object for any structure
    
    Arguments
    ---------
        sequences : Tensor
            Set of sequences
        tags : Tensor
            Set of tags assigned to seqeunces
    
    Returns
    -------
        self : TODO
            TODO

    Notes
    -----
    
    """
    def __init__(self, sequences, tags):
        self.sequences = sequences
        self.tags = tags
        self.lens = get_lengths(self.sequences)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sequences[idx], self.lens[idx], self.tags[idx]