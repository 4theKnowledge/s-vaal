"""
Sampler is used to selectively sample unlabelled dataset for oracle annotation. This module
will contain sampling methods for different types of active learning and their respective heuristics,
including S-VAAL.

TODO:
    - Need to import neural models into module for adversarial sampling routine

@author: Tyler Bikaun
"""

import yaml
import torch
import numpy as np
import random

import unittest
import torch
Tensor = torch.Tensor

from connections import load_config


class Sampler:
    """ sampler """
    def __init__(self, budget: int):
        self.budget = budget

    def _sim_model(self, data: Tensor) -> Tensor:
        """ Simulated model for generating uncertainity scores. Intention
            is to be a placeholder until real models are used and for testing."""
        return torch.rand(size=(data.shape[0],))
        
    def sample_random(self, indices: list) -> Tensor:
        """ Random I.I.D sampling
        Arguments
        ---------
            indices : list
                List of indices corresponding to data samples
        Returns
        -------
            labelled_pool_indices : list
                List of randomly sampled indices w.r.t budget constraint
        """
        # idx = torch.randperm(data.nelement())
        # data_s = data.view(-1)[idx].view(data.size())[:self.budget]
        # return data_s

        budget = len(indices) if self.budget > len(indices) else self.budget        # To ensure that last set of samples doesn't fail on top-k if available indices are LT budget size
        return random.sample(list(indices), k=budget)

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
        top_k = torch.topk(input=uncertainty_scores, k=self.budget, largest=True)
        data_s = torch.index_select(input=data, dim=0, index=top_k.indices)
        # print(f'Uncertainty scores:\n{uncertainty_scores}')
        # print(f'Top K indices: {top_k.indices}')
        # print(data_s)

        return data_s

    def sample_bayesian(self, model, no_models: int, data: Tensor) -> Tensor:
        """ Bayesian sampling (BALD)
        Process: (TODO: flesh out process below)
            1. Generate n-Bayesian networks from randomly sampling from posterior distributions over weights and biases
            2. Compute confidences on unlabelled dataset
            3. Select top-N via BALD metric (TODO: review paper for implementation details)
        
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
        
        threshold = 0.5 # uncertainty threshold

        # Generate set of models by sampling posterior distributions
        models = [model] * no_models


        model_results = dict()
        concat_tensor = torch.tensor([])
        model_no = 1
        for model in models:
            # Get sample uncertainties
            uncertainties = model(data)
            preds = uncertainties.gt(threshold).long()    # int rather than bool

            # print(f'No: {model_no}\n{preds.tolist()}')

            model_results[model_no] = dict()
            model_results[model_no]['Uncertainties'] = uncertainties
            model_results[model_no]['Predictions'] = preds
            model_no += 1

            # add data to concat_tensor for aggregation
            concat_tensor = torch.cat((concat_tensor, preds),0)

        # Aggregate model results
        agg_tensor = torch.sum(concat_tensor.view(-1,10), dim=0)

        # Select top-k set
        top_k = torch.topk(input=agg_tensor, k=self.budget, largest=True)
        data_s = torch.index_select(input=data, dim=0, index=top_k.indices)

        print(f'Aggregate Tensor: {agg_tensor}\nTop_K Tensor Indices: {top_k.indices}\nInput Data:{data}\nData Selection: {data_s}')

        return data_s

    def sample_adversarial(self, svae, discriminator, data, indices, cuda):
        """ Adversarial sampling

        Process:
            1. ...
            2. ...
            3. ...
        
        Arguments
        ---------
            vae : torch model
                VAE model
            discriminator : torch model
                discriminator model
            data : tensor
                Sequence dataset
            cuda : boolean
                GPU flag
            indices : list
                List of indices corresponding to unlabelled set of samples

        Returns
        -------
            querry_pool_indices: int, list
                List of indices corresponding to sorted (top-K) samples to be sampled from
        
        Notes
        -----

        """
        # Code cloned from VAAL
        # TODO: modify for sequence data and associated data structures

        all_preds = []
        for sequences, lengths, _ in data:  # data is (seq, len, tag)

            if torch.cuda.is_available():
                sequences = sequences.cuda()
                lengths = lengths.cuda()

            with torch.no_grad():
                _, _, mean, z = svae(sequences, lengths)
                preds = discriminator(z)    #mean # output should be a flat list of probabilities that the sample is labelled or unlabelled
            
            preds = preds.view(-1)

            preds = preds.cpu().data

            all_preds.extend(preds)
        
        all_preds = torch.stack(all_preds)

        # Need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # Select the points which the discriminator thinks are the most likely to be unlabelled samples
        budget = len(indices) if self.budget > len(indices) else self.budget        # To ensure that last set of samples doesn't fail on top-k if available indices are LT budget size
        print(f'budget: {budget}')
        _, labelled_indices = torch.topk(all_preds, budget)
        labelled_pool_indices = np.asarray(indices)[labelled_indices.numpy()]   # extends the labelled set

        return labelled_pool_indices


class Tests(unittest.TestCase):
    def setUp(self):
        # Init class
        self.sampler = Sampler(budget=10)
        # Init random tensor
        self.data = torch.rand(size=(10,2,2))  # dim (batch, length, features)
        # Params
        self.budget = 18

    # All sample tests are tested for:
    #   1. dims (_, length, features) for input and output Tensors
    #   2. batch size == sample size
    def test_sample_random(self):
        self.assertEqual(self.sampler.sample_random(self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_random(self.data).shape[0], self.sampler.budget)

    def test_sample_least_confidence(self):
        self.assertEqual(self.sampler.sample_least_confidence(model=self.sampler._sim_model, data=self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_least_confidence(model=self.sampler._sim_model, data=self.data).shape[0], self.sampler.budget)

    # def test_sample_bayesian(self):
    #     self.assertEqual(self.sampler.sample_bayesian(model=self.sampler._sim_model, no_models=3, data=self.data).shape[1:], self.data.shape[1:])
    #     self.assertEqual(self.sampler.sample_bayesian(model=self.sampler._sim_model, no_models=3, data=self.data).shape[0], self.sampler.budget)

    # def test_adversarial_sample(self):
        # self.assertEqual(self.sampler.sample_adversarial(self.data).shape[1:], self.data.shape[1:])
        # self.assertEqual(self.sampler.sample_adversarial(self.data).shape[0], self.sampler.budget)

def main():
    
    budget = 8    # amount of TOTAL samples that can be provided to an oracle
    budget = 64    # amount of samples an oracle needs to provide ground truths for

    # Testing functionality
    sampler = Sampler(budget=budget)

    # print('Running method tests')
    unittest.main()


if __name__ == '__main__':
    # Seeds
    config = load_config()
    np.random.seed(config['Train']['seed'])
    torch.manual_seed(config['Train']['seed'])

    main()