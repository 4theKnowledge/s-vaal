"""
Pretraining script for VAE

Process:
    1. Train VAE on X_U
    2. Save pretrained VAE weights
    3. Use pretrained VAE in VAAL architecture

@author: Tyler Bikaun
"""

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
import sys, traceback


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# torch.cuda.empty_cache()


from models import SVAE, Discriminator
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
        
        # Real data
        self.data_name = self.config['Utils'][self.task_type]['data_name']
        self.data_splits = self.config['Utils'][self.task_type]['data_split']
        self.pad_idx = self.config['Utils']['special_token2idx']['<PAD>']
        
        # Test run properties
        self.epochs = self.config['Train']['epochs']
        self.svae_iterations = self.config['Train']['svae_iterations']
        
        self.kfold_xval = False
    
    def _init_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config['Train']['batch_size']

        # Load pre-processed data
        path_data = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'pretrain', 'data.json')
        path_vocab = os.path.join('/home/tyler/Desktop/Repos/s-vaal/data', self.task_type, self.data_name, 'pretrain', 'vocab.json')    # not vocabs
        data = load_json(path_data)
        
        self.vocab = load_json(path_vocab)       # Required for decoding sequences for interpretations. TODO: Find suitable location... or leave be...
        self.vocab_size = len(self.vocab['word2idx'])
        
        self.idx2word = self.vocab['idx2word']
        self.word2idx = self.vocab['word2idx']
        
        self.datasets = dict()
        if self.kfold_xval:
            # Perform k-fold cross-validation
            # Join all datasets and then randomly assign train/val/test
            print('hello')
            
            for split in self.data_splits:
                print(data[split][self.x_y_pair_name])
            
        else:    
            for split in self.data_splits:
                # Access data
                split_data = data[split]
                # print(split_data)
                # Convert lists of encoded sequences into tensors and stack into one large tensor
                split_inputs = torch.stack([torch.tensor(value['input']) for key, value in split_data.items()])
                split_targets = torch.stack([torch.tensor(value['target']) for key, value in split_data.items()])
                # Create torch dataset from tensors
                split_dataset = RealDataset(sequences=split_inputs, tags=split_targets)
                # Add to dictionary
                self.datasets[split] = split_dataset #split_dataloader
                
                # Create torch dataloader generator from dataset
                if split == 'test':
                    self.test_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                if split == 'valid':
                    self.val_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                if split == 'test':
                    self.train_dataloader = DataLoader(dataset=split_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f'{datetime.now()}: Data loaded succesfully')
        
    def _init_svae_model(self):
        self.svae = SVAE(**self.model_config['SVAE']['Parameters'],vocab_size=self.vocab_size).to(self.device)
        self.svae_optim = optim.Adam(self.svae.parameters(), lr=self.model_config['SVAE']['learning_rate'])
        self.svae.train()
        print(f'{datetime.now()}: Initialised SVAE successfully')
    
    def interpolate(self, start, end, steps):
        
        interpolation = np.zeros((start.shape[0], steps+2))
        
        for dim, (s, e) in enumerate(zip(start, end)):
            interpolation[dim] = np.linspace(s, e, steps+2)
            
        return interpolation.T
    
    
    def _idx2word_inf(self, idx, i2w, pad_idx):
        # inf-erence
        sent_str = [str()]*len(idx)
        
        for i, sent in enumerate(idx):
            for word_id in sent:
                if word_id == pad_idx:
                    break
                
                sent_str[i] += i2w[str(word_id.item())] + " "
            sent_str[i] = sent_str[i].strip()
        return sent_str
    
    def _pretrain_svae(self):       
        self._init_data()
        self._init_svae_model()
        
        tb_writer = SummaryWriter(comment=f"pretrain svae {self.data_name}", filename_suffix=f"pretrain svae {self.data_name}")       
        print(f'{datetime.now()}: Training started')
        
        step = 0
        for epoch in range(1, self.config['Train']['epochs']+1, 1):
            for batch_inputs, batch_lengths, batch_targets in self.train_dataloader:
                if torch.cuda.is_available():
                    batch_inputs = batch_inputs.to(self.device)
                    batch_lengths = batch_lengths.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                
                batch_size = batch_inputs.size(0)
                logp, mean, logv, _ = self.svae(batch_inputs, batch_lengths, pretrain=False)
                NLL_loss, KL_loss, KL_weight = self.svae.loss_fn(logp=logp,
                                                                 target=batch_targets,
                                                                 length=batch_lengths,
                                                                 mean=mean,
                                                                 logv=logv,
                                                                 anneal_fn=self.model_config['SVAE']['anneal_function'],
                                                                 step=step,
                                                                 k=self.model_config['SVAE']['k'],
                                                                 x0=self.model_config['SVAE']['x0'])
                svae_loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                self.svae_optim.zero_grad()
                svae_loss.backward()
                self.svae_optim.step()

                tb_writer.add_scalar('Loss/train/KLL', KL_loss, step)
                tb_writer.add_scalar('Loss/train/NLL', NLL_loss, step)
                tb_writer.add_scalar('Loss/train/Total', svae_loss, step)
                tb_writer.add_scalar('Utils/train/KL_weight', KL_weight, step)


                # Increment step after each batch of data
                step += 1
                
            if epoch % 1 == 0:
                print(f'{datetime.now()}: Epoch {epoch} Loss {svae_loss:0.2f} Step {step}')
            
            if epoch % 5 == 0:
                # Perform inference
                self.svae.eval()
                try:
                    samples, z = self.svae.inference(n=2)
                    print(*self._idx2word_inf(samples, i2w=self.idx2word, pad_idx=self.config['Utils']['special_token2idx']['<PAD>']), sep='\n')
                except:
                    traceback.print_exc(file=sys.stdout)
                self.svae.train()
                
        # Save final model
        save_path = os.getcwd() + '/best models/svae.pt'
        torch.save(self.svae.state_dict(), save_path)
        print(f'{datetime.now()}: Model saved')
        
        
        print(f'{datetime.now()}: Training finished')
        
    
if __name__ == '__main__':
    mt = ModularTrainer()
    mt._pretrain_svae()
    
    # mt._init_data()
    # for inputs, lengths, targets in mt.datasets['test']: #test_dataloader
    #     print(inputs)
    #     print(targets)
    #     print(lengths)
    #     break