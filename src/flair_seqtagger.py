"""
Implementation of Flair sequence tagger for CoNLL-2003 state of the art experiments.

Due to the peculiarities of Flair, we need to generate sets of data and write to disk incrementally.
After each write event, we then save these documents for future analysis in their own folders.

Process:
    1. Partition dataset
    2. Put in common directory for Flair sequence taggert
    3. Copy dataset into local results folder for the particular run
    4. Train Flair model
    5. Save model results in run folder 

@author: Tyler Bikaun
"""

from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from flair.data import Corpus
from flair.datasets import CONLL_03, ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List
from flair.models import SequenceTagger

from flair.trainers import ModelTrainer


from utils import load_json, trim_padded_seqs
from connections import load_config
from datetime import datetime

from models import TaskLearner

class TLTrainer:
    def __init__(self):
        self.config = load_config()
        self.model_config = self.config['Models']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.task_type = self.config['Utils']['task_type']
        self.max_sequence_length = self.config['Utils'][self.task_type]['max_sequence_length']
        
        # Real data
        self.data_name = self.config['Utils'][self.task_type]['data_name']
        self.pad_idx = self.config['Utils']['special_token2idx']['<PAD>']
        
        # Test run properties
        self.epochs = self.config['Train']['epochs']
        
        self.labelled_data_path = os.path.join(r'/home/tyler/Desktop/Repos/s-vaal/src/results/10', 'labelled_data.txt')
        self.vocab_path = os.path.join(r'/home/tyler/Desktop/Repos/s-vaal/data/SEQ/conll2003', 'vocabs.json')


    def load_labelled_data(self):
        # Loads DataLoader from disk containing labelled dataset
        # self.labelled_dataloader = torch.load(self.dataloader_path)
        
        with open(self.labelled_data_path, 'r') as fr:
            data = fr.readlines()
        
        print(data)
        
        

    def load_vocab(self):
        # Loads vocabulary data from disk
        vocab_data = load_json(self.vocab_path)
        self.tagset_size = len(vocab_data["tags"])
        self.vocab_size = len(vocab_data["words"])

    def init_task_learner(self):
        """ Initialises models, loss functions, optimisers and sets models to training mode """
        self.task_learner = TaskLearner(hidden_dim=256,
                                        rnn_type='gru',
                                        vocab_size=self.vocab_size,
                                        tagset_size=self.tagset_size,
                                        task_type=self.task_type).to(self.device)
        if self.task_type == 'SEQ':
            self.tl_loss_fn = nn.NLLLoss().to(self.device)

        self.tl_optim = optim.SGD(self.task_learner.parameters(), lr=self.model_config['TaskLearner']['learning_rate'])#, momentum=0, weight_decay=0.1)
        
        # Learning rate scheduler
        # Note: LR likely GT Adam
        # self.tl_sched = optim.lr_scheduler.ReduceLROnPlateau(self.tl_optim, 'min', factor=0.5, patience=10)
        # Training Modes
        self.task_learner.train()
    
        print(f'{datetime.now()}: Initialised Task Learner successfully')
    

    def train(self):
        
        self.load_vocab()
        self.load_labelled_data()
        self.init_task_learner()        
        
        for epoch in range(1, 2, 1):
        
            for sequences, lengths, tags in self.labelled_dataloader:
                
                if torch.cuda.is_available():
                    sequences = sequences.to(self.device)
                    lengths = lengths.to(self.device)
                    tags = tags.to(self.device)
                
                
                tags = trim_padded_seqs(batch_lengths=lengths,
                                        batch_sequences=sequences,
                                        pad_idx=self.pad_idx).view(-1)
                
                self.task_learner.zero_grad()
                self.tl_optim.zero_grad()
                
                tl_preds = self.task_learner(sequences, lengths)
                tl_loss = self.tl_loss_fn(tl_preds, tags)
                tl_loss.backward()
                self.tl_optim.step()
                
            
            print(f'Completed epoch {epoch}')
                
            
            
            
    def train_flair(self):
        
        # Flair Model Initialisation and Training
        # # 1. get the corpus
        # corpus: Corpus = ColumnCorpus(os.path.join(os.getcwd(), 'results', '10'),
        #                               {0: 'text', 1: 'ner'},
        #                               train_file='train.txt',
        #                               test_file='test.txt',
        #                               dev_file='valid.txt',
        #                               column_delimiter=' ')
        
        corpus: Corpus = CONLL_03(base_path=os.path.join(os.getcwd(), 'results', '10'))
        
        corpus.dev_file = 'valid.txt'   # rather than 'dev.txt'
        
        # 2. what tag do we want to predict?
        tag_type = 'ner'
        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        
        # initialize embeddings
        embedding_types: List[TokenEmbeddings] = [
            # GloVe embeddings
            WordEmbeddings('glove'),
            # contextual string embeddings, forward
            PooledFlairEmbeddings('news-forward', pooling='min'),
            # contextual string embeddings, backward
            PooledFlairEmbeddings('news-backward', pooling='min'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        # initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type)

        # initialize trainer
        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        results = trainer.train(os.path.join(os.getcwd(), 'results', '10', 'tagger'),
                                train_with_dev=False,
                                max_epochs=50)
        
        print(results)
        
        
if __name__ == '__main__':
    tlt = TLTrainer()
    
    # tlt.train()
    # tlt.load_labelled_data()
    
    tlt.train_flair()