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
import torch
Tensor = torch.Tensor


class Sampler:
    """ sampler """
    def __init__(self, config, budget: int, sample_size: int):
        self.config = config
        # probably will put these in config in the future
        self.budget = budget
        self.sample_size = sample_size

    def _sim_model(self, data: Tensor) -> Tensor:
        """ Simulated model for generating uncertainity scores. Intention
            is to be a placeholder until real models are used and for testing."""
        return torch.rand(size=(data.shape[0],))
        
    def sample_random(self, data: Tensor) -> Tensor:
        """ Random I.I.D sampling
        Arguments
        ---------
            data : Tensor
                Unlabelled dataset
        Returns
        -------
            data_s : Tensor
                Set of randomly sampled data from unlabelled dataset
        """
        idx = torch.randperm(data.nelement())
        data_s = data.view(-1)[idx].view(data.size())[:self.sample_size]
        return data_s

    def sample_least_confidence(self, model, data: Tensor) -> Tensor:
        """ Least confidence sampling

        Process:
            1. Compute confidences on unlabelled dataset by passing through model
            2. Select top-k
            3. Get top-k indices and subset unlabelled dataset
            4. Return selection
        
        Arguments
        ---------
            data : Tensor
                Unlabelled dataset
            model : 
                Parameterised task learner
        Returns
        -------
            data_s : Tensor
                Set of selectively sampled data from unlabelled dataset
        """
        uncertainty_scores = model(data)
        top_k = torch.topk(input=uncertainty_scores, k=self.sample_size, largest=True)
        data_s = torch.index_select(input=data, dim=0, index=top_k.indices)
        # print(f'Uncertainty scores:\n{uncertainty_scores}')
        # print(f'Top K indices: {top_k.indices}')
        # print(data_s)

        return data_s

    def sample_bayesian(self, model, data: Tensor) -> Tensor:
        """ Bayesian sampling (BALD)
        
        Arguments
        ---------
            data : Tensor
                Unlabelled dataset
            model : 
                Parameterised Bayesian neural network task learner
        Returns
        -------
            data_s : Tensor
                Set of selectively sampled data from unlabelled dataset
        """
        # Generate n-Bayesian networks from randomly sampling from posterior distributions over weights and biases
        # Compute confidences on unlabelled dataset
        # Select top-N via BALD metric (TODO: review paper for implementation details)
        
        return data[:self.sample_size]

    def sample_adversarial(self, vae, discriminator, data, cuda):
        """ Adversarial sampling
        
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
        # Code cloned from VAAL
        # TODO: modify for sequence data and associated data structures

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


class Tests(unittest.TestCase):
    def setUp(self):
        # Init class
        self.sampler = Sampler(config='x', budget=10, sample_size=2)
        # Init random tensor
        self.data = torch.rand(size=(10,10,4))  # dim (batch, length, features)

    # All sample tests are tested for:
    #   1. dims (_, length, features) for input and output Tensors
    #   2. batch size == sample size
    def test_sample_random(self):
        self.assertEqual(self.sampler.sample_random(self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_random(self.data).shape[0], self.sampler.sample_size)

    def test_sample_least_confidence(self):
        self.assertEqual(self.sampler.sample_least_confidence(model=self.sampler._sim_model, data=self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_least_confidence(model=self.sampler._sim_model, data=self.data).shape[0], self.sampler.sample_size)

    # def test_sample_bayesian(self):
    #     self.assertEqual(self.sampler.sample_bayesian(self.data).shape[1:], self.data.shape[1:])
    #     self.assertEqual(self.sampler.sample_bayesian(self.data).shape[0], self.sampler.sample_size)

    # def test_adversarial_sample(self):
    #     self.assertEqual(self.sampler.sample_adversarial(self.data).shape[1:], self.data.shape[1:])
    #     self.assertEqual(self.sampler.sample_adversarial(self.data).shape[0], self.sampler.sample_size)

def main(config):
    
    budget = 500    # amount of TOTAL samples that can be provided to an oracle
    sample_size = 64    # amount of samples an oracle needs to provide ground truths for

    # Testing functionality
    sampler = Sampler(config=config, budget=budget, sample_size=sample_size)

    # print('Running method tests')
    unittest.main()


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)