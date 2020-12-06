"""
Modular training for decoupled variational adversarial active learning
    - Training of TL and VAE/Discriminator are done asynchronously

@author: Tyler Bikaun
"""

from datetime import datetime
import os
from tqdm import tqdm
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
torch.cuda.empty_cache()

from models import TaskLearner, SVAE, Discriminator, Generator
from data import RealDataset
from utils import load_json, trim_padded_seqs
from connections import load_config

from sampler import Sampler

class ModularTrainer(Sampler):
    def __init__(self):
        self.config = load_config()
        self.model_config = self.config['Models']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # Model
        self.task_type = self.config['Utils']['task_type']
        self.max_sequence_length = self.config['Utils'][self.task_type]['max_sequence_length']
        
        self.budget_frac = self.config['Train']['budget_frac']
        self.batch_size = self.config['Train']['batch_size']
        self.data_splits_frac = np.round(np.linspace(self.budget_frac, self.budget_frac*10, num=10, endpoint=True), 2)
        
        
        # Real data
        self.data_name = self.config['Utils'][self.task_type]['data_name']
        self.data_splits = self.config['Utils'][self.task_type]['data_split']
        self.pad_idx = self.config['Utils']['special_token2idx']['<PAD>']
        
        # Test run properties
        self.epochs = self.config['Train']['epochs']
        self.svae_iterations = self.config['Train']['svae_iterations']
        self.dsc_iterations = self.config['Train']['discriminator_iterations']
        self.adv_hyperparam = self.config['Models']['SVAE']['adversarial_hyperparameter']
        
        print('initialised')
    
    def _init_data(self, batch_size=None):
        kfold_xval = False
        
        self.x_y_pair_name = 'seq_label_pairs_enc' if self.data_name == 'ag_news' else 'seq_tags_pairs_enc' # Key in dataset - semantically correct for the task at hand.

        if batch_size is None:
            batch_size = self.config['Train']['batch_size']

        # Load pre-processed data
        path_data = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'data.json')
        path_vocab = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'vocabs.json')
        data = load_json(path_data)
        self.vocab = load_json(path_vocab)       # Required for decoding sequences for interpretations. TODO: Find suitable location... or leave be...
        self.vocab_size = len(self.vocab['words'])  # word vocab is used for model dimensionality setting + includes special characters (EOS, SOS< UNK, PAD)
        self.tagset_size = len(self.vocab['tags'])  # this includes special characters (EOS, SOS, UNK, PAD)

        self.datasets = dict()
        if kfold_xval:
            # Perform k-fold cross-validation
            # Join all datasets and then randomly assign train/val/test
            print('hello')
            
            
            for split in self.data_splits:
                print(data[split][self.x_y_pair_name])
            
        else:    
            for split in self.data_splits:
                # Access data
                split_data = data[split][self.x_y_pair_name]
                # Convert lists of encoded sequences into tensors and stack into one large tensor
                split_seqs = torch.stack([torch.tensor(enc_pair[0]) for key, enc_pair in split_data.items()])
                split_tags = torch.stack([torch.tensor(enc_pair[1]) for key, enc_pair in split_data.items()])
                # Create torch dataset from tensors
                split_dataset = RealDataset(sequences=split_seqs, tags=split_tags)
                # Add to dictionary
                self.datasets[split] = split_dataset #split_dataloader
                
                # Create torch dataloader generator from dataset
                if split == 'test':
                    self.test_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                if split == 'valid':
                    self.val_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f'{datetime.now()}: Data loaded succesfully')
        
        train_dataset = self.datasets['train']
        dataset_size = len(train_dataset)
        
        self.budget = math.ceil(self.budget_frac*dataset_size)
        Sampler.__init__(self, self.budget)
        
        all_indices = set(np.arange(dataset_size))
        k_initial = math.ceil(len(all_indices)*self.budget_frac)
        initial_indices = random.sample(list(all_indices), k=k_initial)
        
        sampler_init = torch.utils.data.sampler.SubsetRandomSampler(initial_indices)
        
        self.labelled_dataloader = DataLoader(train_dataset, sampler=sampler_init, batch_size=self.batch_size, drop_last=True)
        self.val_dataloader = DataLoader(self.datasets['valid'], batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=True, drop_last=False)

        print(f'{datetime.now()}: Dataloaders sizes: Train {len(self.labelled_dataloader)} Valid {len(self.val_dataloader)} Test {len(self.test_dataloader)}')
        return all_indices, initial_indices
    
    def _init_svae_model(self):
        self.svae = SVAE(**self.model_config['SVAE']['Parameters'],vocab_size=self.vocab_size).to(self.device)
        self.svae_optim = optim.Adam(self.svae.parameters(), lr=self.model_config['SVAE']['learning_rate'])
        self.svae.train()
        print(f'{datetime.now()}: Initialised SVAE successfully')
    
    def _init_disc_model(self):
        self.discriminator = Discriminator(**self.model_config['Discriminator']['Parameters']).to(self.device)
        self.dsc_loss_fn = nn.BCELoss().to(self.device)
        self.dsc_optim = optim.Adam(self.discriminator.parameters(), lr=self.model_config['Discriminator']['learning_rate'])
        self.discriminator.train()
        print(f'{datetime.now()}: Initialised Discriminator successfully')
    
    def _init_gen_model(self):
        # Shares same parameters as the discriminator for now. TODO: Update configuration file
        self.generator = Generator(**self.model_config['Discriminator']['Parameters']).to(self.device)
        self.gen_loss_fn = nn.BCELoss().to(self.device)
        self.gen_optim = optim.Adam(self.generator.parameters(), lr=self.model_config['Discriminator']['learning_rate'])
        self.generator.train()
        print(f'{datetime.now()}: Initialised Generator successfully')
    
    def _load_pretrained_model(self):
        self.svae = SVAE(**self.model_config['SVAE']['Parameters'], vocab_size=self.vocab_size).to(self.device)
        # Loads pre-trained SVAE model from disk and modifies
        svae_best_model_path = 'best models/svae.pt'
        self.svae.load_state_dict(torch.load(svae_best_model_path))
        print(self.svae)
        
        # Remove last layer of SVAE
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
            def forward(self, x):
                return x
        
        self.svae.outputs2vocab=Identity()
        print(self.svae)
        
    def _disc_train(self, sequences_l, lengths_l, sequences_u, lengths_u):
        # Train discriminator
        # self.discriminator.zero_grad()
        
        batch_size_l = sequences_l.shape(0)
        batch_size_u = sequences_u.shape(0)
        # Pass through pretrained svae
        with torch.no_grad():
            _, _, _, z_l = self.svae(sequences_l, lengths_l)
            _, _, _, z_u = self.svae(sequences_u, lengths_u)

        # Train discriminator on labelled
        dsc_preds_l = self.discriminator(z_l)
        dsc_real_l = torch.ones(batch_size_l)
        dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)

        # Train discriminator on unlabelled
        dsc_preds_u = self.discriminator(z_u)
        dsc_real_u = torch.zeros(batch_size_u)
        dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)

        if torch.cuda.is_available():
            dsc_real_l = dsc_real_l.to(self.device)
            dsc_real_u = dsc_real_u.to(self.device)

        # Discriminator wants to minimise the loss here
        total_dsc_loss = dsc_loss_l + dsc_loss_u
        self.dsc_optim.zero_grad()
        total_dsc_loss.backward()
        self.dsc_optim.step()
        
        return total_dsc_loss.data.item()
        
    def _gen_train(self, sequences_l, lengths_l, sequences_u, lengths_u):
        # Train Generator
        # self.generator.zero_grad()
        
        batch_size_l = sequences_l.size(0)
        batch_size_u = sequences_u.size(0)
        with torch.no_grad():
            _, _, _, z_l = self.svae(sequences_l, lengths_l)
            _, _, _, z_u = self.svae(sequences_u, lengths_u)
        
        # Adversarial loss - trying to fool the discriminator!
        gen_preds_l = self.discriminator(z_l)
        gen_preds_u = self.discriminator(z_u)
        gen_real_l = torch.ones(batch_size_l)
        gen_real_u = torch.ones(batch_size_u)

        if torch.cuda.is_available():
            dsc_real_l = dsc_real_l.to(self.device)
            dsc_real_u = dsc_real_u.to(self.device)

        # Higher loss = discriminator is having trouble figuring out the real vs fake
        # Generator wants to maximise this loss
        gen_loss_l = self.gen_loss_fn(gen_preds_l, gen_real_l)
        gen_loss_u = self.gen_loss_fn(gen_preds_u, gen_real_u)
        total_gen_loss = gen_loss_l + gen_loss_u

        self.gen_optim.zero_grad()
        total_gen_loss.backward()
        self.gen_optim.step()
        
        return total_gen_loss.data.item()
    
    
    def _train_svaal_pretrained(self):
        # Trains SVAAL using pretrained SVAE and adversarial training routine
        
        
        all_sampled_indices_dict = dict()
        all_indices, initial_indices = self._init_data()
        current_indices = list(initial_indices)
        
        self._load_pretrained_model()   # load pretrained SVAE
        
        print(f'{datetime.now()}: Split regime: {self.data_splits_frac}')
        for split in self.data_splits_frac:
            print(f'{datetime.now()}: Running {split*100:0.0f}% of training dataset')

            # initialise discriminator and generator models for training
            self._init_disc_model()
            self._init_gen_model()
            
            unlabelled_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabelled_sampler = data.sampler.SubsetRandomSampler(unlabelled_indices)
            unlabelled_dataloader = data.DataLoader(self.datasets['train'],
                                                    sampler=unlabelled_sampler,
                                                    batch_size=self.config['Train']['batch_size'],
                                                    drop_last=False)
            print(f'{datetime.now()}: Indices - Labelled {len(current_indices)} Unlabelled {len(unlabelled_indices)} Total {len(all_indices)}')
            # Save indices of X_l, X_u
            all_sampled_indices_dict[str(int(split*100))] = {'Labelled': current_indices, 'Unlabelled': unlabelled_indices}
            # print(all_sampled_indices_dict)
            
            dataloader_l = self.labelled_dataloader
            dataloader_u = unlabelled_dataloader
            dataset_size = len(dataloader_l) + len(dataloader_u)
            train_iterations = dataset_size * self.epochs
            print(f'{datetime.now()}: Dataset size (batches) {dataset_size} Training iterations (batches) {train_iterations}')

            epoch = 1      
            for train_iter in tqdm(range(train_iterations), desc='Training iteration'):
                batch_sequences_l, batch_lengths_l, _ = next(iter(dataloader_l))
                batch_sequences_u, batch_lengths_u, _ = next(iter(dataloader_u))

                if torch.cuda.is_available():
                    batch_sequences_l = batch_sequences_l.to(self.device)
                    batch_lengths_l = batch_lengths_l.to(self.device)
                    batch_sequences_u = batch_sequences_u.to(self.device)
                    batch_length_u = batch_lengths_u.to(self.device)

                batch_size_l = batch_sequences_l.size(0)
                batch_size_u = batch_sequences_u.size(0)

                # Discriminator
                disc_loss = self._disc_train(sequences_l=batch_sequences_l, lengths_l=batch_lengths_l, sequences_u=batch_sequences_u, lengths_u=batch_length_u)

                # Generator
                gen_loss = self._gen_train(sequences_l=batch_sequences_l, lengths_l=batch_lengths_l, sequences_u=batch_sequences_u, lengths_u=batch_length_u)
            
                if (train_iter >0) & (train_iter % dataset_size == 0):
                    train_iter_str = f'{datetime.now()}: Epoch {epoch} - Losses ({self.task_type}) | Disc {disc_loss:0.2f} | Gen {gen_loss:0.2f} | Learning rates: ...'
                    print(train_iter_str)
                    epoch += 1
                    
            
            # Adversarial sample
            sampled_indices, _, _ = self.sample_adversarial(svae=self.svae,
                                                            discriminator=self.discriminator,
                                                            data=dataloader_u,
                                                            indices=unlabelled_indices,
                                                            cuda=True)
            
            # Update indices -> Update dataloaders
            current_indices = list(current_indices) + list(sampled_indices)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(current_indices)
            self.labelled_dataloader = DataLoader(self.datasets['train'], sampler=sampler, batch_size=self.batch_size, drop_last=True)
    
    def _train_svaal(self):
        
        all_sampled_indices_dict = dict()
        
        all_indices, initial_indices = self._init_data()
        current_indices = list(initial_indices)
        
        print(f'{datetime.now()}: Split regime: {self.data_splits_frac}')
        for split in self.data_splits_frac:
            print(f'{datetime.now()}: Running {split*100:0.0f}% of training dataset')

            self._init_svae_model()
            self._init_disc_model()
            
            unlabelled_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabelled_sampler = data.sampler.SubsetRandomSampler(unlabelled_indices)
            unlabelled_dataloader = data.DataLoader(self.datasets['train'],
                                                    sampler=unlabelled_sampler,
                                                    batch_size=self.config['Train']['batch_size'],
                                                    drop_last=False)
            print(f'{datetime.now()}: Indices - Labelled {len(current_indices)} Unlabelled {len(unlabelled_indices)} Total {len(all_indices)}')
            # Save indices of X_l, X_u
            all_sampled_indices_dict[str(int(split*100))] = {'Labelled': current_indices, 'Unlabelled': unlabelled_indices}
            # print(all_sampled_indices_dict)
            
            
            dataloader_l = self.labelled_dataloader
            dataloader_u = unlabelled_dataloader
            
            
            dataset_size = len(dataloader_l) + len(dataloader_u)
            train_iterations = dataset_size * self.epochs
            print(f'{datetime.now()}: Dataset size (batches) {dataset_size} Training iterations (batches) {train_iterations}')

            step = 0
            epoch = 1      
            for train_iter in tqdm(range(train_iterations), desc='Training iteration'):
                batch_sequences_l, batch_lengths_l, _ = next(iter(dataloader_l))
                batch_sequences_u, batch_lengths_u, _ = next(iter(dataloader_u))

                if torch.cuda.is_available():
                    batch_sequences_l = batch_sequences_l.to(self.device)
                    batch_lengths_l = batch_lengths_l.to(self.device)
                    batch_sequences_u = batch_sequences_u.to(self.device)
                    batch_length_u = batch_lengths_u.to(self.device)

                batch_size_l = batch_sequences_l.size(0)
                batch_size_u = batch_sequences_u.size(0)

                # SVAE Step
                for i in range(self.svae_iterations):
                    logp_l, mean_l, logv_l, z_l = self.svae(batch_sequences_l, batch_lengths_l)
                    NLL_loss_l, KL_loss_l, KL_weight_l = self.svae.loss_fn(
                                                                    logp=logp_l,
                                                                    target=batch_sequences_l,
                                                                    length=batch_lengths_l,
                                                                    mean=mean_l,
                                                                    logv=logv_l,
                                                                    anneal_fn=self.model_config['SVAE']['anneal_function'],
                                                                    step=step,
                                                                    k=self.model_config['SVAE']['k'],
                                                                    x0=self.model_config['SVAE']['x0'])

                    logp_u, mean_u, logv_u, z_u = self.svae(batch_sequences_u, batch_lengths_u)
                    NLL_loss_u, KL_loss_u, KL_weight_u = self.svae.loss_fn(
                                                                    logp=logp_u,
                                                                    target=batch_sequences_u,
                                                                    length=batch_lengths_u,
                                                                    mean=mean_u,
                                                                    logv=logv_u,
                                                                    anneal_fn=self.model_config['SVAE']['anneal_function'],
                                                                    step=step,
                                                                    k=self.model_config['SVAE']['k'],
                                                                    x0=self.model_config['SVAE']['x0'])
                    # VAE loss
                    svae_loss_l = (NLL_loss_l + KL_weight_l * KL_loss_l) / batch_size_l
                    svae_loss_u = (NLL_loss_u + KL_weight_u * KL_loss_u) / batch_size_u

                    # Adversarial loss - trying to fool the discriminator!
                    dsc_preds_l = self.discriminator(z_l)   # mean_l
                    dsc_preds_u = self.discriminator(z_u)   # mean_u
                    dsc_real_l = torch.ones(batch_size_l)
                    dsc_real_u = torch.ones(batch_size_u)

                    if torch.cuda.is_available():
                        dsc_real_l = dsc_real_l.to(self.device)
                        dsc_real_u = dsc_real_u.to(self.device)

                    # Higher loss = discriminator is having trouble figuring out the real vs fake
                    # Generator wants to maximise this loss
                    adv_dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)
                    adv_dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)
                    adv_dsc_loss = adv_dsc_loss_l + adv_dsc_loss_u

                    total_svae_loss = svae_loss_u + svae_loss_l + self.adv_hyperparam * adv_dsc_loss
                    self.svae_optim.zero_grad()
                    total_svae_loss.backward()
                    self.svae_optim.step()

                    # Sample new batch of data while training adversarial network
                    if i < self.svae_iterations - 1:
                        batch_sequences_l, batch_lengths_l, _ =  next(iter(dataloader_l))
                        batch_sequences_u, batch_length_u, _ = next(iter(dataloader_u))

                        if torch.cuda.is_available():
                            batch_sequences_l = batch_sequences_l.to(self.device)
                            batch_lengths_l = batch_lengths_l.to(self.device)
                            batch_sequences_u = batch_sequences_u.to(self.device)
                            batch_length_u = batch_length_u.to(self.device)
                    
                    # Increment step
                    step += 1

                # SVAE train_iter loss after iterative cycle
                # self.tb_writer.add_scalar('Loss/SVAE/train/Total', total_svae_loss, train_iter)

                # Discriminator Step
                for j in range(self.dsc_iterations):

                    with torch.no_grad():
                        _, mean_l, _, z_l = self.svae(batch_sequences_l, batch_lengths_l)
                        _, mean_u, _, z_u = self.svae(batch_sequences_u, batch_lengths_u)

                    dsc_preds_l = self.discriminator(z_l)
                    dsc_preds_u = self.discriminator(z_u)

                    dsc_real_l = torch.ones(batch_size_l)
                    dsc_real_u = torch.zeros(batch_size_u)

                    if torch.cuda.is_available():
                        dsc_real_l = dsc_real_l.to(self.device)
                        dsc_real_u = dsc_real_u.to(self.device)

                    # Discriminator wants to minimise the loss here
                    dsc_loss_l = self.dsc_loss_fn(dsc_preds_l, dsc_real_l)
                    dsc_loss_u = self.dsc_loss_fn(dsc_preds_u, dsc_real_u)
                    total_dsc_loss = dsc_loss_l + dsc_loss_u
                    self.dsc_optim.zero_grad()
                    total_dsc_loss.backward()
                    self.dsc_optim.step()

                    # Sample new batch of data while training adversarial network
                    if j < self.dsc_iterations - 1:
                        # TODO: strip out unnecessary information
                        batch_sequences_l, batch_lengths_l, _ =  next(iter(dataloader_l))
                        batch_sequences_u, batch_length_u, _ = next(iter(dataloader_u))

                        if torch.cuda.is_available():
                            batch_sequences_l = batch_sequences_l.to(self.device)
                            batch_lengths_l = batch_lengths_l.to(self.device)
                            batch_sequences_u = batch_sequences_u.to(self.device)
                            batch_length_u = batch_length_u.to(self.device)
            
                if (train_iter >0) & (train_iter % dataset_size == 0):
                    train_iter_str = f'{datetime.now()}: Epoch {epoch} - Losses ({self.task_type}) | SVAE {total_svae_loss:0.2f} | Disc {total_dsc_loss:0.2f} | Learning rates: '
                    print(train_iter_str)
                    
                    epoch += 1
                    
            
            # Adversarial sample
            sampled_indices, _, _ = self.sample_adversarial(svae=self.svae,
                                                      discriminator=self.discriminator,
                                                      data=dataloader_u,
                                                      indices=unlabelled_indices,
                                                      cuda=True)
            
            # Update indices -> Update dataloaders
            current_indices = list(current_indices) + list(sampled_indices)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(current_indices)
            self.labelled_dataloader = DataLoader(self.datasets['train'], sampler=sampler, batch_size=self.batch_size, drop_last=True)
    
    
    def _init_tl_model(self):
        """ Initialises models, loss functions, optimisers and sets models to training mode """
        self.task_learner = TaskLearner(**self.model_config['TaskLearner']['Parameters'],
                                        vocab_size=self.vocab_size,
                                        tagset_size=self.tagset_size,
                                        task_type=self.task_type).to(self.device)
        if self.task_type == 'SEQ':
            self.tl_loss_fn = nn.NLLLoss().to(self.device)
        if self.task_type == 'CLF':
            self.tl_loss_fn = nn.CrossEntropyLoss().to(self.device)

        self.tl_optim = optim.SGD(self.task_learner.parameters(), lr=self.model_config['TaskLearner']['learning_rate'])#, momentum=0, weight_decay=0.1)
        
        # Learning rate scheduler
        # Note: LR likely GT Adam
        # self.tl_sched = optim.lr_scheduler.ReduceLROnPlateau(self.tl_optim, 'min', factor=0.5, patience=10)
        # Training Modes
        self.task_learner.train()
    
        print(f'{datetime.now()}: Initialised Task Learner successfully')
    
    def _train_tl(self, dataloader_l):
        
        self.init_tl_model()
        
        train_iterations = len(dataloader_l) * (self.epochs+1)
        
        for train_iter in tqdm(range(train_iterations), desc='Training iteration'):
            batch_sequences_l, batch_lengths_l, batch_tags_l =  next(iter(dataloader_l))

            if torch.cuda.is_available():
                batch_sequences_l = batch_sequences_l.to(self.device)
                batch_lengths_l = batch_lengths_l.to(self.device)
                batch_tags_l = batch_tags_l.to(self.device)
            
            # Strip off tag padding and flatten
            # Don't do sequences here as its done in the forward pass of the seq2seq models
            batch_tags_l = trim_padded_seqs(batch_lengths=batch_lengths_l,
                                            batch_sequences=batch_tags_l,
                                            pad_idx=self.pad_idx).view(-1)

            # Task Learner Step
            self.tl_optim.zero_grad()
            tl_preds = self.task_learner(batch_sequences_l, batch_lengths_l)
            tl_loss = self.tl_loss_fn(tl_preds, batch_tags_l)
            tl_loss.backward()
            self.tl_optim.step()
            
    
if __name__ == '__main__':
    mt = ModularTrainer()
    # mt._init_data()
    # mt._train_svaal()
    mt._train_svaal_pretrained()