"""
Sampler is used to selectively sample unlabelled dataset for oracle annotation. This module
will contain sampling methods for different types of active learning and their respective heuristics,
including S-VAAL.

@author: Tyler Bikaun
"""

import yaml
import torch
import numpy as np

import unittest


# Code copied from VAAL - TODO: modify for sequence data
class Sampler:
    """ Adversary sampler """
    def __init__(self, config, budget: int, sample_size: int):
        self.config = config
        # probably will put these in config in the future
        self.budget = budget
        self.sample_size = sample_size

        
    def sample(self, vae, discriminator, data, cuda):
        """ Selective sampling algorithm
        
        Arguments
        ---------
            vae : torch model
                VAE model
            discriminator : torch model
                discriminator model
            data : tensor
                Image data
            cuda : boolean
                GPU flag
        Returns
        -------
            querry_pool_indices: int, list
                List of indices corresponding to sorted (top-K) samples to be sampled from
        """
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

    def sample1(self):
        return 


class Tests(unittest.TestCase, Sampler):

    def test_svaal_sample(self):
        sampler = Sampler(config='x', budget=10, sample_size=2)
        self.assertTrue(sampler.sample_size > 0)


def main(config):
    
    
    budget = 500    # amount of TOTAL samples that can be provided to an oracle
    sample_size = 64    # amount of samples an oracle needs to provide ground truths for

    # Testing functionality
    sampler = Sampler(config=config, budget=budget, sample_size=sample_size)

    # print('Running method tests')
    # unittest.main()


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)