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
from datetime import date
import os
import sys, traceback

import torch
Tensor = torch.Tensor


class DataPreparation:
    """ Utility functions for preparing sequence labelling datasets """
    def __init__(self, config):
        self.utils_config = config['Utils']
        self.task_type = self.utils_config['task_type']
        self.data_name = self.utils_config[self.task_type]['data_name']
        self.min_occurence = self.utils_config[self.task_type]['min_occurence']
        self.special_tokens = self.utils_config['special_token2idx']
        self.date = date.today().strftime('%d-%m-%Y')
        self.max_seq_len = self.utils_config[self.task_type]['max_sequence_length']
        self.x_y_pair_name = 'seq_label_pairs' if self.data_name == 'ag_news' else 'seq_tags_pairs' # Key in dataset - semantically correct for the task at hand.
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'

        if self.task_type == 'NER':
            print('NER TASK :)!')
            self._load_data()
            self._process_data_ner()

        elif self.task_type=='CLF':
            print('CLF TASK! :)')
            self._load_data()
            self._process_data_clf()

        else:
            raise ValueError

    def _read_txt(self, path):
        """ """
        f = open(path, 'r')
        data = f.readlines()

        if self.task_type == 'NER':
            # remove DOCSTART (this is specific to conll2003 original formatting)
            if self.data_name == 'conll2003':
                data = [line for line in data if 'DOCSTART' not in line]
            f.close()
        elif self.task_type == 'CLF':
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
        # print(data[:5])
        return data

    def _load_data(self):
        """ Loads data for each split and combines into a dictionary for downstream tasks """
        if self.utils_config[self.task_type]['data_split']:
            self.dataset = dict()
            for split in self.utils_config[self.task_type]['data_split']:
                self.dataset[split] = dict()
                # Read text documents
                if self.data_name == 'conll2003':
                    self.dataset[split]['corpus'] = self._read_txt(os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{split}.txt'))
                elif self.data_name == 'ag_news':
                    self.dataset[split]['corpus'] = self._read_csv(os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{split}.csv'))
                else:
                    raise ValueError
        else:
            # No splits, single corpora -> need to split into test-train-valid (TODO: future work)
            pass

    def _process_data_clf(self):
        """ Controller for processing text classification data"""
        
        # seq_class_pairs
        
        for split, data in self.dataset.items():
            print(f'{split} - {len(data["corpus"])}')
            self._prepare_sequences(split=split, data=data)

            # Trim and pad sequences
            self._trim_sequences(split=split)
            self._pad_sequences(split=split)
            self._add_special_tokens(split=split)

            if split == 'train':
                print('Building vocabularies and mappings from training data')
                self._build_vocabs()
                self._word2idx()
                self._idx2word()
                self._tag2idx()
                self._idx2tag()
            
            self.convert_sequences(split=split)

        # Save results (add datetime and counts)
        self._save_json(path=os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{self.data_name}_{self.date}.json'), data=self.dataset)

    def _process_data_ner(self):
        """ Controller for processing named entity recognition (sequence) data """
        for split, data in self.dataset.items():
            # Convert corpora into key-value pairs of sequences-tags
            # Need to seperate words and tags before trimming and padding
            # a tad of duplication, but all good.
            self._prepare_sequences(split=split, data=data)

            # Trim and pad sequences
            self._trim_sequences(split=split)
            self._pad_sequences(split=split)
            self._add_special_tokens(split=split)

            if split == 'train':
                print('Building vocabularies and mappings from training data')
                self._build_vocabs()
                self._word2idx()
                self._idx2word()
                self._tag2idx()
                self._idx2tag()

            self.convert_sequences(split=split)
        
        # Save results (add datetime and counts)
        self._save_json(path=os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{self.data_name}_{self.date}.json'), data=self.dataset)

    def _prepare_sequences(self, split : str, data):
        """ Converts corpus into sequence-tag tuples.
        Notes
        -----
            - Currently works for NER (CoNLL-2003) and CLF (AG NEWS)
            - Extend for POS
        """
        corpus = data['corpus']
        if self.data_name == 'conll2003':
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
                if self.data_name == 'conll2003':
                    sequence = [token.split(delimiter)[0] for token in doc]
                    tags = [token.split(delimiter)[-1] for token in doc]
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

        # Remove special_tokens (these are added explicitly later)
        word_list = [word for word in word_list if word not in list(self.special_tokens.keys())]

        word_freqs = {}
        # Get word frequencies
        for word in word_list:
            if not word_freqs.get(word, False):
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
        # print(word_freqs)
        
        # Get set of frequent words over minimum occurence
        word_list_keep = list()
        for word, freq in word_freqs.items():
            if self.min_occurence <= freq:
                # keep word
                word_list_keep.append(word)

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

        # Save vocabularies to disk
        vocabs = {'words': self.vocab_words, 'tags': self.vocab_tags}
        self._save_json(path=os.path.join(self.utils_config[self.task_type]['data_root_path'], f'{self.data_name}_vocabs_{self.date}.json'),data=vocabs)
        
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
            seq, tags = pair    # tag for CLF, tags for NER
            self.dataset[split][self.x_y_pair_name][idx] = (seq[:self.max_seq_len], tags[:self.max_seq_len])

    def _pad_sequences(self, split: str):
        """ Pads sequences up to the maximum allowable length """
        for idx, pair in self.dataset[split][self.x_y_pair_name].items():
            seq, tags = pair
            if len(seq) < self.max_seq_len:
                # probably a better way to do this, but comprehension is easy. TODO: fix dodgy code!
                seq = seq + [self.pad_token for _ in range(self.max_seq_len - len(seq))]
                if self.task_type == 'NER':
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
            if self.task_type == 'NER':
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


def main(config):
    """"""
    # do something someday
    DataPreparation(config)


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
    main(config)