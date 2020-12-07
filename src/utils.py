"""
Utilities for various stages of the modelling process including data preparation.

TODO:
- Add data preprocessor for sequence tasks (NER - BIO, and POS)
    - NER -> CoNLL2003, OntoNotes-5.0
    - POS -> CoNLL2003(TODO: confirm), PTB
- Add data preprocessor for text classification (CLF)

@author: Tyler Bikaun
"""

import yaml
import csv
import json
from itertools import groupby
import itertools
import re
import math
from datetime import date, datetime
import os
import sys, traceback
import unittest
from nltk.tokenize import word_tokenize
from collections import defaultdict
import io

from connections import load_config

import torch
Tensor = torch.Tensor


class DataPreparation:
    """ Utility functions for preparing sequence labelling datasets """
    def __init__(self):
        config = load_config()

        self.utils_config = config['Utils']
        self.task_type = self.utils_config['task_type']
        self.data_name = self.utils_config[self.task_type]['data_name']
        self.min_occurence = self.utils_config[self.task_type]['min_occurence']
        self.special_tokens = self.utils_config['special_token2idx']
        self.date = date.today().strftime('%d-%m-%Y')
        self.max_seq_len = self.utils_config[self.task_type]['max_sequence_length']
        self.x_y_pair_name = 'seq_label_pairs' if self.task_type == 'CLF' else 'seq_tags_pairs' # Key in dataset - semantically correct for the task at hand.
        self.pad_token = '<PAD>'
        self.sos_token = '<START>'
        self.eos_token = '<STOP>'

        print(f'{datetime.now()}: Building {self.data_name.upper()} data for {self.task_type.upper()} task')
        if self.task_type == 'SEQ':
            self._load_data()
            self._process_data_ner()
            self._process_pretrain_data_ner()
                
        elif self.task_type=='CLF':
            self._load_data()
            self._process_data_clf()
            
        else:
            raise ValueError

    def _read_txt(self, path):
        f = open(path, 'r')
        data = f.readlines()

        if self.task_type == 'SEQ':
            if self.data_name == 'conll2003':
                # CoNLL-2003 (NER)
                # remove DOCSTART (this is specific to conll2003 original formatting)
                data = [line for line in data if 'DOCSTART' not in line]

            if self.data_name == 'ontonotes-5.0' or 'bbn':
                data = [line for line in data]
            f.close()

        elif self.task_type == 'CLF':
            # Currently no CLF data that needs text processing
            pass

        return data

    def _read_csv(self, path):
        """ Reads data in .CSV format
        
        Arguments
        ---------
            path : str
                Path to .csv file location
        Returns
        -------
            data : list
                List of tuples corresponding to Xn, y pairs/triples
        """
        data = list()
        with open(path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # data = list()
            corpus_str = ''
            for row in csv_reader:
                if self.data_name == 'ag_news':
                    # ag_news has col 1 - label, col 2 - headline, col 3 - article text
                    # and has no column headers. For this dataset we concatenate the headline to the article text
                    corpus_str += f'{row[1] + " " +row[2]}\t{row[0]}\n'

        data = [line for line in corpus_str.split('\n') if line]    # if statement gets rid of empty strings
        return data

    def _load_data(self):
        """ Loads data for each split and combines into a dictionary for downstream tasks """
        if self.utils_config[self.task_type]['data_split']:
            self.dataset = dict()
            for split in self.utils_config[self.task_type]['data_split']:
                self.dataset[split] = dict()
                # Read text documents
                if self.data_name == 'conll2003' or 'ontonotes-5.0' or 'bbn':
                    self.dataset[split]['corpus'] = self._read_txt(os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{split}.txt'))
                elif self.data_name == 'ag_news':
                    self.dataset[split]['corpus'] = self._read_csv(os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{split}.csv'))
                else:
                    raise ValueError
        else:
            # No splits, single corpora -> need to split into test-train-valid (TODO: future work)
            pass

    def _process_data_clf(self):
        pass
        # # Trim and pad sequences
        # self._trim_sequences(split=split)
        # self._add_special_tokens(split=split)
        # self._pad_sequences(split=split)

        # if split == 'train':
        #     print('Building vocabularies and mappings from training data')
        #     self._build_vocabs()
        #     self._word2idx()
        #     self._idx2word()
        #     self._tag2idx()
        #     self._idx2tag()
        #     self._save_vocabs() # do this after word2idx etc as it allows us to save them into the same json as vocabs

        # self.convert_sequences(split=split)
        
        # # Save results (add datetime and counts)
        # self._save_json(path=os.path.join(self.utils_config[self.task_type]['data_root_path'], f'data.json'), data=self.dataset)

    def _process_data_ner(self):
        """ Controller for processing named entity recognition (sequence) data """
        for split, data in self.dataset.items():
            # Convert corpora into key-value pairs of sequences-tags
            # Need to seperate words and tags before trimming and padding
            # a tad of duplication, but all good.
            self._prepare_sequences(split=split, data=data)

            # Trim and pad sequences
            self._trim_sequences(split=split)
            self._add_special_tokens(split=split)
            self._pad_sequences(split=split)

            if split == 'train':
                print('Building vocabularies and mappings from training data')
                self._build_vocabs()
                self._word2idx()
                self._idx2word()
                self._tag2idx()
                self._idx2tag()
                self._save_vocabs() # do this after word2idx etc as it allows us to save them into the same json as vocabs

            self.convert_sequences(split=split)
        
        # Save results (add datetime and counts)
        self._save_json(path=os.path.join(self.utils_config[self.task_type]['data_root_path'], f'data.json'), data=self.dataset)



    def _process_pretrain_data_ner(self):
        
        data_out = defaultdict(dict)
        
        print(f'{datetime.now()}: Generating pretraining datasets')
        for split, data in self.dataset.items():
            data = data['corpus']
            assert self.data_name == 'conll2003'    # only developed for conll2003 atm
            
            docs = [list(group) for k, group in groupby(data, lambda x: len(x) == 1) if not k]
            
            if split == 'train':
                w2c = dict()
                w2i = dict()
                i2w = dict()
                
                for st, idx in self.utils_config['special_token2idx'].items():
                    i2w[idx] = st
                    w2i[st] = idx
                    
                for idx, doc in enumerate(docs):
                    # conll2003 needs to be split on tab before tokenization
                    doc = " ".join([token.split()[0] for token in doc])
                    # words = word_tokenize(doc)
                    words = doc.split()
                    
                    # Trim based on sequence length
                    # This should be -1 but as it needs to be the same size as the normal vocab, it's -2
                    words = words[:self.max_seq_len-2]  # -2 for SOS, EOS tags
                    
                    
                    for word in words:
                        if word in w2c.keys():
                            w2c[word] += 1
                        else:
                            w2c[word] = 1
                
                
                for w, c in w2c.items():
                    if c > self.min_occurence and w not in self.utils_config['special_token2idx'].keys():
                        i2w[len(w2i)] = w
                        w2i[w] = len(w2i)
                
                assert len(w2i) == len(i2w)
                
                print(f'{datetime.now()}: Vocab of {len(w2i)} keys created')
                
                vocab = {"word2idx": w2i, "idx2word": i2w}
                
                with io.open(os.path.join(self.utils_config[self.task_type]['data_root_path'], 'pretrain', 'vocab.json'), 'w') as vocab_file:
                    json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)
                    # vocab_file.write(data.encode('utf8', 'replace'))

            data_out[split] = dict()
            
            for idx, doc in enumerate(docs):
                id = len(data_out[split])
                data_out[split][id] = dict()
                
                doc = " ".join([token.split()[0] for token in doc])
                
                # words = word_tokenize(doc)
                words = doc.split()
                
                input = ['<START>'] + words
                input = input[:self.max_seq_len]
                
                target = words[:self.max_seq_len-1]
                target = target + ['<STOP>']
                
                assert len(input) == len(target)
                
                length = len(input)
                
                input.extend(['<PAD>'] * (self.max_seq_len-length))
                target.extend(['<PAD>'] * (self.max_seq_len-length))
                
                input = [w2i.get(w, w2i['<UNK>']) for w in input]
                target = [w2i.get(w, w2i['<UNK>']) for w in target]
                
                data_out[split][id]['input'] = input
                data_out[split][id]['target'] = target
                data_out[split][id]['length'] = length

            with io.open(os.path.join(self.utils_config[self.task_type]['data_root_path'], 'pretrain', 'data.json'), 'w') as data_file:
                json.dump(data_out, data_file, ensure_ascii=False, indent=4)


    def _prepare_sequences(self, split : str, data):
        """ Converts corpus into sequence-tag tuples.
        Notes
        -----
            - Currently works for NER (CoNLL-2003) and CLF (AG NEWS)
            - Extend for POS
        """
        print(f'{datetime.now()}: Preparing sequences')
        corpus = data['corpus']
        if self.data_name == 'conll2003' or 'ontonotes-5.0' or 'bbn':
            docs = [list(group) for k, group in groupby(corpus, lambda x: len(x) == 1) if not k]
            
        elif self.data_name == 'ag_news':
            docs = corpus

        self.dataset[split][self.x_y_pair_name] = list()
        data = dict()
        # Split docs into sequences and tags
        doc_count = 0
        delimiter = '\t' if self.data_name == 'ag_news' else ' '
        for doc in docs:
            try:
                if self.data_name == 'conll2003' or 'ontonotes-5.0' or 'bbn':
                    sequence = [token.split(delimiter)[0] for token in doc]
                    tags = [token.split(delimiter)[-1] for token in doc]
                    tags = [tag.replace('\n','') for tag in tags]
                    # print(tags)
                    data[doc_count] = (sequence, tags)
                    
                elif self.data_name == 'ag_news':
                    sequence = doc.split(delimiter)[0].split()    # split seq from seq-tag string and then split on white space for naive tokenization
                    tag = [doc.split(delimiter)[1]]
                    data[doc_count] = (sequence, tag)
                doc_count += 1
                
            except:
                print(f'Unable to process document: {doc}')
                traceback.print_exc(file=sys.stdout)

        self.dataset[split][self.x_y_pair_name] = data

    def _build_vocabs(self, split='train'):
        """ Builds vocabularies off of words and tags. These are built from training data so out of vocabulary tokens
        will be marked as <UNK> when being converted into numerical vectors. """

        # Get list of words in corpus
        word_list = list(itertools.chain.from_iterable([doc.split() for doc in [" ".join(seq) for seq, tag in self.dataset[split][self.x_y_pair_name].values()]]))

        print(f'Total number of tokens in training corpus: {len(word_list)}')

        # Remove special_tokens (these are added explicitly later)
        word_list = [word for word in word_list if word not in list(self.special_tokens.keys())]
        # print(word_list)

        word_freqs = dict()
        # Get word frequencies
        for word in word_list:
            if not word_freqs.get(word, False):
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
        # print(word_freqs)
        
        # Get set of frequent words over minimum occurence
        word_list_keep = list()
        word_list_notkeep = list()
        for word, freq in word_freqs.items():
            if self.min_occurence < freq:
                # keep word
                word_list_keep.append(word)
            else:
                word_list_notkeep.append(word+'\n')
                
        if self.min_occurence > 0:
            with open(os.path.join(self.utils_config[self.task_type]['data_root_path'], 'nonvocab_words.txt'), 'w') as fw:
                fw.writelines(word_list_notkeep)

        print(f'Word list sizes - Original: {len(word_freqs.keys())} - Trimmed: {len(word_list_keep)}')
        
        # Build word and tag vocabularies
        # comprehensions are a bit nasty... but effective!
        self.vocab_words = word_list_keep
        tag_list = list(itertools.chain.from_iterable([doc.split() for doc in [" ".join(tag) for seq, tag in self.dataset[split][self.x_y_pair_name].values()]]))
        # Remove special_tokens (these are added explicitly later)
        tag_list = [tag for tag in tag_list if tag not in list(self.special_tokens.keys())]
        
        self.vocab_tags = list(set(tag_list))
        
        # Add special_tokens to vocabs
        self.vocab_words = list(self.special_tokens.keys()) + self.vocab_words
        self.vocab_tags = list(self.special_tokens.keys()) + self.vocab_tags
        print(f'Size of vocabularies - Word: {len(self.vocab_words)} Tag: {len(self.vocab_tags)}')

    def _save_vocabs(self):
        # Save vocabularies to disk
        vocabs = {'words': self.vocab_words, 'tags': self.vocab_tags, 'word2idx': self.word2idx, 'idx2word': self.idx2word}
        self._save_json(path=os.path.join(self.utils_config[self.task_type]['data_root_path'], 'vocabs.json'),data=vocabs)
        
    def _word2idx(self):
        """ Built off of training set - out of vocab tokens are <UNK>"""
        self.word2idx = {word:idx for idx, word in enumerate(self.vocab_words)}
        # add special tokens to mapping
        self.word2idx = {**self.special_tokens, **self.word2idx, }

    def _idx2word(self):
        """ Built off of training set - out of vocab tokens are <UNK> """
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}

    def _tag2idx(self):
        """ Built off of training set - out of vocab tokens are <UNK>"""
        self.tag2idx = {tag:idx for idx, tag in enumerate(self.vocab_tags)}

    def _idx2tag(self):
        """ Built off of training set - out of vocab tokens are <UNK> """
        self.idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}

    def _trim_sequences(self, split: str):
        """ Trims sequences to the maximum allowable length """
        for idx, pair in self.dataset[split][self.x_y_pair_name].items():
            seq, tags = pair    # tag for CLF, tags for SEQ
            self.dataset[split][self.x_y_pair_name][idx] = (seq[:self.max_seq_len-2], tags[:self.max_seq_len-2])    # -2 for SOS, EOS tags

    def _pad_sequences(self, split: str):
        """ Pads sequences up to the maximum allowable length """
        for idx, pair in self.dataset[split][self.x_y_pair_name].items():
            seq, tags = pair
            if len(seq) < self.max_seq_len:
                # probably a better way to do this, but comprehension is easy. TODO: fix dodgy code!
                seq = seq + [self.pad_token for _ in range(self.max_seq_len - len(seq))]
                if self.task_type == 'SEQ':
                    tags = tags + [self.pad_token for _ in range(self.max_seq_len - len(tags))]
                    self.dataset[split][self.x_y_pair_name][idx] = (seq, tags)
                else:
                    # Leave tag alone
                    self.dataset[split][self.x_y_pair_name][idx] = (seq, tags)

    def _add_special_tokens(self, split: str):
        """ Adds special tokens such as <EOS>, <SOS> onto sequences """
        for idx, pair in self.dataset[split][self.x_y_pair_name].items():
            seq, tags = pair
            seq = [self.sos_token] + seq + [self.eos_token]
            if self.task_type == 'SEQ':
                tags = [self.sos_token] + tags + [self.eos_token]
                self.dataset[split][self.x_y_pair_name][idx] = (seq, tags)
            else:
                # Leave tag alone
                self.dataset[split][self.x_y_pair_name][idx] = (seq, tags)
                
    def convert_sequences(self, split: str):
        """
        Converts sequences of tokens and tags to their indexed forms for each split in the dataset
        
        Note: any word in the sequence that isn't in the vocab will be replaced with <UNK>
        TODO: investigate how this might impact the SVAE word_dropout methodology """

        # If word or tag is not in the sequence, change with <UNK>
        # unsure if the output tags need to be changed? I assume not as the output tags are known. TODO: verify logic.

        self.dataset[split][f'{self.x_y_pair_name}_enc'] = dict()  # enc -> integer encoded pairs
        for idx, pair in self.dataset[split][self.x_y_pair_name].items():
            seq, tags = pair
            # Sequences
            seq_enc = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in seq]
            # Tags
            tags_enc = [self.tag2idx.get(tag, self.word2idx['<UNK>']) for tag in tags]
            # print(tags)
            # print(tags_enc)

            self.dataset[split][f'{self.x_y_pair_name}_enc'][idx] = (seq_enc, tags_enc)

    def normalise(self):
        pass

    def _save_json(self, path: str, data: dict):
        with open(path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

# Misc functions below
def trim_padded_seqs(batch_lengths: Tensor, batch_sequences: Tensor, pad_idx: int) -> Tensor:
    """ Takes a batch of sequences and trims similar to pack padded sequence method 
    
    Arguments
    ---------
        batch_lengths : Tensor
            Batch of sequence lengths
        batch_tags : Tensor
            Batch of sequences
        pad_idx : Int
            Integer mapped to padding special token

    Returns
    -------
        batch_tags : Tensor
            Sorted and trimmed batch of sequence tags

    """
    # Get max length of longest sequence in batch so it can be used to filter tags
    sorted_lengths, _ = torch.sort(batch_lengths, descending=True)      # longest seq is at index 0

    longest_seq = sorted_lengths[0].data.cpu().numpy()
    longest_seq_len = longest_seq[longest_seq != pad_idx][0]       # remove padding
    
    # Strip off as much padding as possible similar to (variable length sequences via pack padded methods)
    batch_sequences = torch.stack([tags[:longest_seq_len] for tags in batch_sequences])

    assert batch_sequences.is_cuda

    return batch_sequences

def to_var(x: Tensor) -> Tensor:
    """ Converts object to variable mounted on GPU """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def load_json(path: str) -> dict:
    """ Loads JSON file from disk 
    
    Arguments
    ---------
        path : str
            Path to JSON file on disk
    
    Returns
    -------
        data : dict
            Dictionary of JSON file
    """
    with open(path, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data

def get_lengths(sequences: Tensor) -> Tensor:
    """ Calculates lengths of sequences 
    
    Arguments
    ---------
        sequences : Tensor
            Set of sequences.
    Returns
    -------
        lengths : Tensor
            Set of sequence lengths
    """
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    return lengths

def split_data(dataset: Tensor, splits: tuple) -> Tensor:
    """ Partitions data into different sets 
    
    Arguments
    ---------
        dataset : Tensor
            Tensor of data.
        splits : tuple
            Tuple of floats indicating ordered splits
    
    Returns
    -------
        dataset : Tensor
            Ordered set of dataset subset objects corresponding to splits 

    Notes
    -----
        random_split can have its generator fixed to be deterministic for reproducible results.
    """
    assert sum(list(splits)) == 1.0

    if len(splits) == 2:
        split_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(math.floor(len(dataset)*splits[0])),
                                                                                int(math.ceil(len(dataset)*splits[1]))])
        return split_dataset[0], split_dataset[1]
    elif len(splits) == 3:
        # TODO: figure out how to ensure that the three splits have the same total samples as the input dataset...
        split_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(len(dataset)*splits[0]),
                                                                                int(len(dataset)*splits[1]),
                                                                                int(len(dataset)*splits[2])])
        return split_dataset[0], split_dataset[1], split_dataset[2]
    else:
        raise ValueError

def prepare_for_embedding():
    """ Prepares sequences for Flair embedding """
    from flair.data import Sentence
    
    text = 'Hello my name is John Snow!'
    
    sentence_e = Sentence(text)
    
    print(embeddings.embed(sentence_e))
    
def main():
    DataPreparation()
    

if __name__ == '__main__':
    # Seeds
    main()