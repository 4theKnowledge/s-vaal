"""
Utilities for various stages of the modelling process including data preparation.

@author: Tyler Bikaun
"""

import yaml
import torch

Tensor = torch.Tensor

class DataPreparation:
    """ Utility functions for preparing sequence labelling datasets """
    def __init__(self):
        pass

    def _save(self):
        """
        """
        pass

def trim_padded_tags(batch_lengths: Tensor, batch_tags: Tensor, pad_idx: int) -> Tensor:
    """ Takes a batch of sequences and tags and trims similar to pack padded sequence method 
    
    Arguments
    ---------
        batch_lengths : Tensor
            Batch of sequence lengths
        batch_tags : Tensor
            Batch of sequence tags
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
    batch_tags = torch.stack([tags[:longest_seq_len] for tags in batch_tags])

    return batch_tags

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