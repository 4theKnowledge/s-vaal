"""
Main script which orchestrates active learning cycles.

@author: Tyler Bikaun
"""

import yaml
import random
import numpy as np

class ActiveLearner:

    def __init__(self):
        pass


def main(config):
    
    # parameters of data and active learning set-up
    num_images = 0
    num_val = 0
    initial_budget = 0

    # ---- Copied from vaal ----
    all_indices = set(np.arange(num_images))
    val_indices = random.sample(all_indices, num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)
    
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