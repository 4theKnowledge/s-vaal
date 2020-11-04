"""
Utilities for various stages of the modelling process.

@author: Tyler Bikaun
"""

import yaml
import torch

class Utils:
    def __init__(self):
        pass

def to_var(self, x):
    """ Converts object to variable mounted on GPU """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def main():
    """"""
    # do something someday
    pass


if __name__ == '__main__':
    main()