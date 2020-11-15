"""
Main script which orchestrates active learning.

@author: Tyler Bikaun
"""

import yaml
import random
import numpy as np

import torch

class ActiveLearner:

    def __init__(self):
        self.initial_budget_frac = 0.10 # fraction of samples that AL starts with
        self.oracle_sample_size = 64    # number of samples that can be selected at each iteration
        self.data_splits = np.round(np.linspace(self.initial_budget_frac, 1, num=10, endpoint=True), 1)


    def learn(self):
        """ Performs the active learning cycle """
        
        # Split indicates how much data the al algorithm is allowed to sample for
        # e.g. if the total sample size if 1000 and the split is 10% then only 100 samples can be given to an oracle

        # Total number of samples available (100%)
        num_samples = 1000

        for split in self.data_splits:
            # partition full dataset to split maximum
            num_split_samples = num_samples*split
            num_samples_l = num_split_samples*self.initial_budget_frac
            num_samples_u = num_split_samples - num_samples_l

            al_round_no = 0
            for al_round in 

            
            # do some stuff
            # add samples into labelled set after annotation and remove from unlabelled set
            num_samples_l += self.oracle_sample_size
            num_samples_u -= self.oracle_sample_size
            print(f'X_L: {num_samples_l} X_U: {num_samples_u} Data remaining: {num_split_samples-num_samples_l} Oracle Sample Size: {self.oracle_sample_size}')


def main(config):
    
    al = ActiveLearner()
    al.learn()


    # parameters of data and active learning set-up
    num_images = 10
    num_val = 1
    initial_budget = 5

    # ---- Copied from vaal ----

    # Create indices against entire dataset and then split for val (X_v, y_v)/train (X_U, X_L)
    all_indices = set(np.arange(num_images))
    val_indices = random.sample(all_indices, num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)
    # print(all_indices, val_indices)

    # 
    initial_indices = random.sample(list(all_indices), initial_budget)
    # print(initial_indices)
    # sampler = data.sampler.SubsetRandomSampler(initial_indices)
    # val_sampler = data.sampler.SubsetRandomSampler(val_indices)
    
    
    # dataset with labels available
    # querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    # val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)
    
    return


    
    cuda = cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []

    # Each split of data

    for split in splits:
        # Need to retrain all the models on the new images
        # re initialise and retrain the models
        task_model = vgg.vgg16_vn(num_classes=num_classes)
        vae = model.VAE(latent_dim)
        discriminator = model.Discriminator(latent_dim)

        unlabelled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabelled_sampler = data.sampler.SubsetRandomSampler(unlabelled_indices)
        unlabelled_dataloader = data.DataLoader(train_dataset, sampler=unlabelled_sampler, batch_size=batch_size, drop_last=False)

        # train the models on current data and returns best final accuracy
        acc, vae, discriminator = solver.train(querry_dataloader,
                                                val_dataloader,
                                                task_model,
                                                vae,
                                                discriminator,
                                                unlabelled_dataloader)

        print(f'Final accuracy is {acc*100}%')
        accuracies.append(acc)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

        torch.save(accuracies, os.path.join(out_path, log_name))

if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    # Seeds
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed'])

    main(config)