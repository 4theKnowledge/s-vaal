"""
Utilities for various stages of the modelling process including data preparation.

TODO:
- Add data preprocessor for NER (BIO) and POS
    - NER -> CoNLL2003
    - POS -> CoNLL2003(TODO: confirm), PTB

@author: Tyler Bikaun
"""

import yaml
import json
from itertools import groupby
import itertools
import re

import torch
Tensor = torch.Tensor


class DataPreparation:
    """ Utility functions for preparing sequence labelling datasets """
    def __init__(self, config):
        self.utils_config = config['Utils']
        self.task_type = self.utils_config['task_type']
        self.special_tokens = self.utils_config['special_token2idx']
        self.min_occurence = self.utils_config['min_occurence']

        if self.task_type=='NER':
            print('NER LIFE :)!')
            self._load_data()
            
            self._prepare_sequences()

            self._save_json(self.utils_config['data_root_path']+f'\CoNLL2003.json', self.dataset)

            self._build_vocabs()

            self._word2idx()
            self._idx2word()

            self._tag2idx()
            self._idx2tag()

        elif self.task_type == 'POS':
            pass
        else:
            raise ValueError

    def _read_txt(self, path):
        f = open(path, 'r')
        text = f.readlines()

        # remove DOCSTART
        text = [line for line in text if 'DOCSTART' not in line]        # DOCSTART is specific to CoNLL2003 original formatting
        f.close()
        return text

    def _save_json(self, path, data):
        with open(path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def _load_data(self):
        if self.utils_config['data_split']:
            self.dataset = dict()
            for split in self.utils_config['data_split']:
                self.dataset[split] = dict()
                # Read text documents
                self.dataset[split]['corpus'] = self._read_txt(self.utils_config['data_root_path']+f'\{split}.txt')
        else:
            # No splits, single corpora
            # need to split into test-train-valid
            # TODO: future work
            pass

    def _prepare_sequences(self):
        """ Converts corpus into sequence-tag tuples.
        TODO:
            - Currently works for CoNLL2003 NER copora
            - Extend for POS 
        """

        for split, data in self.dataset.items():
            corpus = data['corpus']
            docs = [list(group) for k, group in groupby(corpus, lambda x: len(x) == 1) if not k]

            self.dataset[split]['kv_pairs'] = list()
            data = list()
            # Split docs into sequences and tags
            for doc in docs:
                sequence = [token.split()[0] for token in doc]  #  if 'DOCSTART' not in token
                tags = [token.split()[-1] for token in doc]
                data.append((sequence, tags))
            self.dataset[split]['kv_pairs'] = data

    def _build_vocabs(self):
        """ Built off of training set - out of vocab tokens will be marked as <UNK>
        
        Two vocabularies - word and tag

        Minimum occurence (frequency) cut-off is implemented to keep models reasonable sized (CURRENTLY NOT IMPLEMENTED... will need to do enumeration if used)
        """
        split = 'train'

        # Get list of words in corpus
        word_list = list(itertools.chain.from_iterable([doc.split() for doc in [" ".join(seq) for seq, tag in self.dataset[split]['kv_pairs']]]))

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
        self.vocab_tags = list(set(list(itertools.chain.from_iterable([doc.split() for doc in [" ".join(tag) for seq, tag in self.dataset[split]['kv_pairs']]]))))
        print(f'Size of vocabularies - Word: {len(self.vocab_words)} Tag: {len(self.vocab_tags)}')
        
    def _word2idx(self):
        """ Built off of training set - out of vocab tokens are <UNK>"""
        self.word2idx = {word:idx+len(self.special_tokens) for idx, word in enumerate(self.vocab_words)}
        # add special tokens to mapping
        self.word2idx = {**self.special_tokens, **self.word2idx, }
        # print(self.word2idx)

    def _idx2word(self):
        """ Built off of training set - out of vocab tokens are <UNK> """
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}

    def _tag2idx(self):
        """ Built off of training set - out of vocab tokens are <UNK>"""
        self.tag2idx = {tag:idx for idx, tag in enumerate(self.vocab_tags)}
        print(self.tag2idx)

    def _idx2tag(self):
        """ Built off of training set - out of vocab tokens are <UNK> """
        self.idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        print(self.idx2tag)

    
    def normalise(self):
        pass

    def _save(self):
        """
        """
        pass


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
    longest_seq = sorted_lengths[0].data.numpy()
    longest_seq_len = longest_seq[longest_seq != pad_idx][0]       # remove padding
    
    # Strip off as much padding as possible similar to (variable length sequences via pack padded methods)
    batch_sequences = torch.stack([tags[:longest_seq_len] for tags in batch_sequences])

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