"""
Sampler is used to selectively sample unlabelled dataset for oracle annotation. This module
will contain sampling methods for different types of active learning and their respective heuristics,
including S-VAAL.

TODO:
    - Need to import neural models into module for adversarial sampling routine
    - Abstract sampling method and write conditionals to treat different methods rather than calling each function individually...

@author: Tyler Bikaun
"""

import yaml
import torch
import numpy as np
import random
import statistics

import unittest
import torch
Tensor = torch.Tensor

from connections import load_config


class Sampler:
    """ Sampler """
    def __init__(self, budget: int):
        self.budget = budget
        
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
        budget = len(indices) if self.budget > len(indices) else self.budget        # To ensure that last set of samples doesn't fail on top-k if available indices are LT budget size
        return random.sample(list(indices), k=budget)

    def sample_least_confidence(self, model, data, indices) -> Tensor:
        """ Least confidence sampling

        Process:
            1. Compute confidences on unlabelled dataset by passing through model. These are 
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
        # Perform inference on unlabelled samples and select top-k based on least confidence (highest uncertainty)

        all_preds = list()

        for sequences, lengths, _ in data:
            
            if torch.cuda.is_available():
                sequences = sequences.cuda()
                lengths = lengths.cuda()
            
            with torch.no_grad():
                preds = model(sequences, lengths)
            
            # TODO: WIP

        
        # uncertainty_scores = model(data)
        # top_k = torch.topk(input=uncertainty_scores, k=self.budget, largest=True)
        # data_s = torch.index_select(input=data, dim=0, index=top_k.indices)
        # print(f'Uncertainty scores:\n{uncertainty_scores}')
        # print(f'Top K indices: {top_k.indices}')
        # print(data_s)

        return data_s

    def sample_max_norm_logp(self, model, data, indices) -> Tensor:
        """ Samples with Maximum Normalized Log-Probability (MNLP) heuristic (Shen et al., 2018) 
        
        Notes
        -----
        TODO: Needs QA to ensure that the log probabilities are implemented correctly. log_p = log(pred)
        """
        

        all_preds = list()
        for sequences, lengths, _ in data:

            if torch.cuda.is_available():
                sequences = sequences.cuda()
                lengths = lengths.cuda()
            
            with torch.no_grad():
                preds = model(sequences, lengths)

            # Need to unpack each sequence of predictions and calculate their normalised log-probability
            # DO SOMETHING HERE
            # preds = f_nlp(preds)

            preds = preds.view(-1)
            preds = preds.cpu().data

            all_preds.extend(preds)

        all_preds = torch.stack(all_preds)

        all_preds *= -1

        budget = len(indices) if self.budget > len(indices) else self.budget
        print(f'budget: {budget}')
        _, labelled_indices = torch.topk(all_preds, budget)
        labelled_pool_indices = np.asarray(indices)[labelled_indices.numpy()]

        return labelled_pool_indices

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

    def sample_adversarial(self, svae, discriminator, data, indices, pretrain):
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
            labelled_pool_indices: int, list
                List of indices corresponding to sorted (top-K) samples to be sampled from
            preds : list
                List of floats correpsonding to predictions disriminator
        """

        all_preds = list()
        for sequences, lengths, _ in data:  # data is (seq, len, tag)

            if torch.cuda.is_available():
                sequences = sequences.cuda()
                lengths = lengths.cuda()

            with torch.no_grad():
                z = svae(sequences, lengths, pretrain)
                
                preds = discriminator(z)    #mean # output should be a flat list of probabilities that the sample is labelled or unlabelled
                # print(preds)
                
            preds = preds.view(-1)

            preds = preds.cpu().data

            all_preds.extend(preds)
        
        # Need to convert each tensor float into float dtype in all_preds
        all_preds_stats = [pred.item() for pred in all_preds]
        
        print(f'Preds: Mean {statistics.mean(all_preds_stats)} Median {statistics.median(all_preds_stats)} Std {statistics.stdev(all_preds_stats)} Min {min(all_preds_stats)} Max {max(all_preds_stats)}')
        
        all_preds_stats_dict = {"mean": statistics.mean(all_preds_stats),
                               "median": statistics.median(all_preds_stats),
                               "stdev": statistics.stdev(all_preds_stats),
                               "min": min(all_preds_stats),
                               "max": max(all_preds_stats)}
        
        all_preds = torch.stack(all_preds)

        # Need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # Select the points which the discriminator thinks are the most likely to be unlabelled samples
        budget = len(indices) if self.budget > len(indices) else self.budget        # To ensure that last set of samples doesn't fail on top-k if available indices are LT budget size
        print(f'Data Sampled: {budget}')

        preds_topk, labelled_indices = torch.topk(all_preds, budget)    # Returns topk values and their indices
        labelled_pool_indices = np.asarray(indices)[labelled_indices.numpy()]   # extends the labelled set

        return labelled_pool_indices, preds_topk, all_preds_stats_dict


if __name__ == '__main__':
    
    # Checking that random sampler works...
    sampler = Sampler(budget=10)
    indices = list(range(0,100,1))
    sampled_indices = sampler.sample_random(indices)
    print(sampled_indices)
    print(len(sampled_indices))