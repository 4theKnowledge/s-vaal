"""
Sampler is used to selectively sample unlabelled dataset for oracle annotation. This module
will contain sampling methods for different types of active learning and their respective heuristics.

@author: Tyler Bikaun
"""

import yaml
import torch
import numpy as np


# Code copied from VAAL - TODO: modify for sequence data
class Sampler:
    """ Adversary sampler """
    def __init__(self, budget):
        self.budget = budget
        
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


def main(config):
    """"""
    # do something someday... like initialise and test the methods herein
    pass


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)