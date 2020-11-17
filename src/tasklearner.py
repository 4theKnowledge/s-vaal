"""
Contains model initialisation procedures and test functionality for task learner.
There are two task learner configurations: 1. Text classifiation (CLF) and 2. Sequence tagging (NER)

TODO:
    - Rename task type as CLF and SEQ rather than CLF and NER... makes more general for other SEQ tasks like POS
    - Add bidirectionality, GRU, RNN, multi-layers

@author: Tyler Bikaun
"""

# Imports
import yaml
import numpy as np
import unittest

from data import DataGenerator
from utils import to_var, trim_padded_seqs, split_data

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
Tensor = torch.Tensor


class TaskLearner(nn.Module):
    """ Initialises a task learner for either text classification or sequence labelling 
    
    Arguments
    ---------
        embedding_dim : int
            Size of embedding dimension.
        hidden_dim : int
            Size of hiddend dimension of rnn model
        vocab_size : int
            Size of input word vocabulary.
        tagset_size : int
            Size of output tag space. For CLF this will be 1, for NER this will be n.
        task_type : str
            Task type of the task learner e.g. CLF for text classification or NER for named entity recognition
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, task_type: str):
        super(TaskLearner, self).__init__()

        self.task_type = task_type  # CLF - text classification; NER - named entity recognition

        self.rnn_type = 'gru'

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # TODO: Implement padding_idx=self.pad_idx
        
        if self.rnn_type == 'gru':
            rnn = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn = nn.LSTM
        elif self.rnn_type == 'rnn':
            rnn = nn.RNN
        else:
            raise ValueError
        
        # Sequence tagger (TODO: implement more advanced sequence taggers and options)
        self.rnn = rnn(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        if self.task_type == 'NER':
            # Linear layer that maps hidden state space from rnn to tag space
            self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=tagset_size)

        if self.task_type == 'CLF':
            self.drop = nn.Dropout(p=0.5)
            self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, batch_sequences: Tensor, batch_lengths: Tensor) -> Tensor:
        """
        Forward pass through Task Learner

        Arguments
        ---------
            batch_sequences : Tensor
                Batch of sequences
            batch_lengths : Tensor
                Batch of sequence lengths

        Returns
        -------
            tag_scores : Tensor
                Batch of predicted tag scores
        """
        input_embeddings = self.word_embeddings(batch_sequences)

        # Sort and pack padded sequence for variable length LSTM
        sorted_lengths, sorted_idx = torch.sort(batch_lengths, descending=True)
        batch_sequences = batch_sequences[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input=input_embeddings,
                                                        lengths=sorted_lengths.data.tolist(),
                                                        batch_first=True)

        rnn_out, _ = self.rnn(packed_input)

        # Unpack padded sequence
        padded_outputs = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        if self.task_type == 'NER':
            # Project into tag space
            tag_space = self.hidden2tag(padded_outputs.view(-1, padded_outputs.size(2)))
            tag_scores = F.log_softmax(tag_space, dim=1)
            return tag_scores

        if self.task_type == 'CLF':
            output = self.drop(padded_outputs)
            output = self.hidden2tag(output)
            tag_scores = torch.sigmoid(output)
            return tag_scores


class TaskLearnerTest(unittest.TestCase):
    def setUp(self):
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parameters
        self.no_seqs = 16
        self.max_seq_len = 4
        self.embedding_dim = 8
        self.hidden_dim = 8
        self.no_class_clf = 4
        self.no_class_seq = 8
        self.epochs = 10
        self.pad_idx = 0    # TODO: Confirm or link to config

        # DataGenerator
        self.datagen = DataGenerator(config=config)
        self.sequences, self.lengths = self.datagen.build_sequences(no_sequences=self.no_seqs,
                                                                    max_sequence_length=self.max_seq_len)
        self.sequences = self.sequences.to(self.device)
        self.lengths = self.lengths.to(self.device)

        self.vocab = self.datagen.build_vocab(self.sequences)
        self.vocab_size = len(self.vocab)

        self.dataset_clf = self.datagen.build_sequence_classes(self.sequences, self.lengths)
        self.dataset_seq = self.datagen.build_sequence_tags(self.sequences, self.lengths)

    def train(self, epochs, pad_idx, model, dataset, loss_fn, optim, task_type):
        """ Training routine for task learners

        Arguments
        ---------
            model : TODO
                Task learner torch model
            dataset : TODO
                Dataset generator
            loss_fn :TODO
                Model loss function
            optim : TODO
                Model optimiser 
        Returns
        -------
            loss : float
                Loss correpsonding to last epoch in training routine 
        """
        for _ in range(epochs):
            for batch_sequences, batch_lengths, batch_labels in dataset:
                
                # print(batch_sequences.shape, batch_lengths.shape, batch_labels.shape)
                
                if torch.cuda.is_available():
                    batch_sequences = batch_sequences.cuda()
                    batch_lengths = batch_lengths.cuda()
                    batch_labels = batch_labels.cuda()
                model.zero_grad()

                # Strip off tag padding (similar to variable length sequences via pack padded methods)
                batch_labels = trim_padded_seqs(batch_lengths=batch_lengths,
                                                batch_sequences=batch_labels,
                                                pad_idx=pad_idx).view(-1)
                # Forward pass through model
                scores = model(batch_sequences, batch_lengths)
                if task_type == 'CLF':
                    batch_labels = batch_labels.view(-1,1)
                # Calculate loss and backpropagate error through model
                loss = loss_fn(scores, batch_labels)
                loss.backward()
                optim.step()

            # print(f'Epoch: {epoch} Loss: {loss}')
        return loss, scores

    def test_tl_clf_train(self):
        tl_clf = TaskLearner(embedding_dim=self.embedding_dim,
                                hidden_dim=self.hidden_dim,
                                vocab_size=self.vocab_size,
                                tagset_size=self.no_class_clf,
                                task_type='CLF').to(self.device)
        loss_fn_clf = nn.CrossEntropyLoss().to(self.device)
        optim_clf = optim.SGD(tl_clf.parameters(), lr=0.1)
        tl_clf.train()
        
        # Get last epoch loss value and scores Tensor
        loss_clf, scores_clf = self.train(epochs=self.epochs,
                                            pad_idx=self.datagen.pad_idx,
                                            model=tl_clf,
                                            dataset=self.dataset_clf,
                                            loss_fn=loss_fn_clf,
                                            optim=optim_clf,
                                            task_type='CLF')
        # Check pred score shape
        self.assertEqual(scores_clf.shape, (self.no_seqs, self.no_class_clf, 1), msg="Predicted scores are the incorrect shape")
        # Check loss
        self.assertTrue(isinstance(loss_clf.item(), float), msg="Loss in not producing a float output")

    def test_tl_seq_train(self):
        tl_seq = TaskLearner(embedding_dim=self.embedding_dim,
                                hidden_dim=self.hidden_dim,
                                vocab_size=self.vocab_size,
                                tagset_size=self.no_class_seq,
                                task_type='NER').to(self.device)
        loss_fn_seq = nn.NLLLoss().to(self.device)
        optim_seq = optim.SGD(tl_seq.parameters(), lr=0.1)
        tl_seq.train()
        
        # Get last epoch loss value and scores Tensor
        loss_seq, scores_seq = self.train(epochs=self.epochs,
                                            pad_idx=self.datagen.pad_idx,
                                            model=tl_seq,
                                            dataset=self.dataset_seq,
                                            loss_fn=loss_fn_seq,
                                            optim=optim_seq,
                                            task_type='NER')
        # Check loss
        self.assertTrue(isinstance(loss_seq.item(), float), msg="Loss in not producing a float output")
        # Check pred score shape
        self.assertEqual(scores_seq.shape, (self.no_seqs*self.max_seq_len, self.no_class_seq), msg="Predicted scores are the incorrect shape")

def main(config):
    # Run tests
    unittest.main()


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