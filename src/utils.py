"""
Utilities for various stages of the modelling process including data preparation.

TODO:
- Add data preprocessor for NER (BIO) and POS
    - NER -> CoNLL2003
    - POS -> PTB

@author: Tyler Bikaun
"""

import yaml
import torch

Tensor = torch.Tensor

class DataPreparation:
    """ Utility functions for preparing sequence labelling datasets """
    def __init__(self):
        pass

    def prepare_ner(self):
        """ """
        pass

    def prepare_pos(self):
        """ """
        pass

    def word2idx(self):
        pass

    def idx2word(self):
        pass

    def build_vocab(self):
        pass

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
    pass


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
    main(config)