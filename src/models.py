"""
Contains model initialisation procedures and test functionality

Contains model initialisation procedures and test functionality for task learner.
There are two task learner configurations: 1. Text classifiation (CLF) and 2. Sequence tagging (SEQ)

TODO:
    - Add bidirectionality, GRU, RNN, multi-layers

@author: Tyler Bikaun
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
Tensor = torch.Tensor

from data import DataGenerator
from utils import to_var, trim_padded_seqs, split_data
from connections import load_config

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List


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
            Size of output tag space. For CLF this will be 1, for SEQ this will be n.
        task_type : str
            Task type of the task learner e.g. CLF for text classification or SEQ (named entity recognition, part of speech tagging)
    """
    def __init__(self, hidden_dim: int,
                 rnn_type: str,
                 vocab_size: int,
                 tagset_size: int,
                 task_type: str):
        super(TaskLearner, self).__init__()

        self.task_type = task_type
        self.rnn_type = rnn_type
        self.bidirectional = True
        self.num_layers = 2
        self.num_directions = 2 if self.bidirectional else 1

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        
        # self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # TODO: Implement padding_idx=self.pad_idx
        
        
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            PooledFlairEmbeddings('news-forward', pooling='min'),
            PooledFlairEmbeddings('news-backward', pooling='min')]
    
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
        self.embeddings = embeddings
        self.embedding_dim : int = self.embeddings.embedding_length
        
        
        if self.rnn_type == 'gru':
            rnn = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn = nn.LSTM
        elif self.rnn_type == 'rnn':
            rnn = nn.RNN
        else:
            raise ValueError
        
        # Sequence tagger
        self.rnn = rnn(input_size=self.embedding_dim,
                       hidden_size=hidden_dim,
                       num_layers=self.num_layers,
                       dropout=0.0 if self.num_layers == 1 else 0.5,
                       bidirectional=self.bidirectional,
                       batch_first=True)
        
        if self.task_type == 'SEQ':
            # Linear layer that maps hidden state space from rnn to tag space
            self.hidden2tag = nn.Linear(in_features=hidden_dim * self.num_directions, out_features=tagset_size)

        if self.task_type == 'CLF':
            # COME BACK LATER...
            self.drop = nn.Dropout(p=0.5)
            self.hidden2tag = nn.Linear(in_features=hidden_dim * self.num_directions, out_features=1)

    
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
        
        input_embeddings = self.embeddings.embed(batch_sequences)
        # input_embeddings = self.word_embeddings(batch_sequences)
        
        

        # Sort and pack padded sequence for variable length RNN
        sorted_lengths, sorted_idx = torch.sort(batch_lengths, descending=True)
        batch_sequences = batch_sequences[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input=input_embeddings,
                                                        lengths=sorted_lengths.data.tolist(),
                                                        batch_first=True)

        rnn_out, hidden = self.rnn(packed_input)

        # Unpack padded sequence
        padded_outputs = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        if self.task_type == 'SEQ':
            # Project into tag space
            tag_space = self.hidden2tag(padded_outputs.view(-1, padded_outputs.size(2)))
            tag_scores = F.log_softmax(tag_space, dim=1)
            return tag_scores

        if self.task_type == 'CLF':
            output = self.drop(padded_outputs)
            output = self.hidden2tag(output)
            tag_scores = torch.sigmoid(output)
            return tag_scores            


class SVAE(nn.Module):
    """ Sequence based Variational Autoencoder"""
    def __init__(self, embedding_dim, hidden_dim, rnn_type, num_layers, bidirectional, latent_size, word_dropout, embedding_dropout, vocab_size: int):
        super(SVAE, self).__init__()
        config = load_config()
        utils_config = config['Utils']
        
        # Misc
        task_type = config['Utils']['task_type']
        self.max_sequence_length = utils_config[task_type]['max_sequence_length']
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.pretrain = config['Train']['pretrain']

        # Specical tokens and vocab
        self.pad_idx = utils_config['special_token2idx']['<PAD>']
        self.eos_idx = utils_config['special_token2idx']['<STOP>']
        self.sos_idx = utils_config['special_token2idx']['<START>']
        self.unk_idx = utils_config['special_token2idx']['<UNK>']
        self.vocab_size = vocab_size #+ len(utils_config['special_token2idx'])
                
        # RNN settings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.z_dim = latent_size
        
        # Embedding initialisation
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # RNN type specification
        # TODO: Future implementation will include transformer/reformer models rather than these.
        if self.rnn_type == 'gru':
            rnn = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn = nn.LSTM
        elif self.rnn_type == 'rnn':
            rnn = nn.RNN
        else:
            raise ValueError()
        
        # Initialise encoder-decoder RNNs (these are identical)
        self.encoder_rnn = rnn(input_size=self.embedding_dim,
                               hidden_size=self.hidden_dim, 
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.decoder_rnn = rnn(input_size=self.embedding_dim,
                               hidden_size=self.hidden_dim, 
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        # Hidden factor is used for expanding dimensionality if bidirectionality and multi-layer functionality is used
        self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers
        
        # Initialisation of FC layers
        # These map from the encoder to the latent space
        self.hidden2mean = nn.Linear(self.hidden_dim * self.hidden_factor, self.z_dim)
        self.hidden2logv = nn.Linear(self.hidden_dim * self.hidden_factor, self.z_dim)
        self.z2hidden = nn.Linear(self.z_dim, self.hidden_dim * self.hidden_factor)
        self.outputs2vocab = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.vocab_size)
        
        # Initialise partial loss function
        self.NLL = nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')   # TODO: Review arguments for understanding

    def forward(self, input_sequence: Tensor, length: Tensor, pretrain: bool) -> Tensor:
        """ 
        Performs forward pass through SVAE model

        Arguments
        ---------
            input_sequence : Tensor
                Batch of input sequences
            length : Tensor
                batch of input sequence lengths
        
        Returns
        -------
            logp :  Tensor
                Log posterior over tag space log(p(x|theta)) (TODO: confirm notation)
            mean : Tensor
                Gaussian mean
            logv : Tensor
                Gaussian variance
            z : Tensor
                SVAE latent space         
        """
        
        if pretrain is None:
            pretrain = self.pretrain
        
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)   # trick for packed padding
        input_sequence = input_sequence[sorted_idx]
        
        # ENCODER
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self._encode(packed_input)
        
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_dim * self.hidden_factor)
        else:
            # .squeeze() -> Returns a tensor with all the dimensions of input of size 1 removed.
            # print(f'hidden shape before squeeze {hidden.shape}')
            #             hidden = hidden.squeeze()   # doesn't work? gives wrong dimension down stream... must be due to their data format or bidirection/n_layers? TODO: test.
            # print(f'hidden shape after squeeze {hidden.shape}')
            pass

        # Reparameterisation trick!
        # z, mean, logv, std = self.reparameterise(hidden, batch_size)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.z_dim]))   # z_dim = latent_size
        z = z * std + mean

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            # hidden = hidden.unsqueeze(0)
            pass
        
        if pretrain:
            # Pretrained model doesn't have output2vocab layer and only requires the latent space
            return z
        else:
            # DECODER
            if self.word_dropout_rate > 0:
                prob = torch.rand(input_sequence.size())

                if torch.cuda.is_available():
                    prob = prob.cuda()

                prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1

                decoder_input_sequence = input_sequence.clone()
                decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
                input_embedding = self.embedding(decoder_input_sequence)

            input_embedding = self.embedding_dropout(input_embedding)
            packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
            outputs, _ = self._decode(packed_input, hidden)
            
            # Process outputs
            # Unpack padded sequence
            padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
            padded_outputs = padded_outputs.contiguous()
            _, reversed_idx = torch.sort(sorted_idx)
            padded_outputs = padded_outputs[reversed_idx]
            b, s, _ = padded_outputs.size()

            # Project outputs to vocab
            # e.g. project hidden state into label space...
            logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
            logp = logp.view(b, s, self.embedding.num_embeddings)

            return logp, mean, logv, z

    def kl_anneal_fn(self, anneal_fn: str, step: int, k: float, x0: int):
        """ KL annealing is used to slowly modify the impact of KL divergence in the loss 
            function for the VAE (Bowman et al., 2015)
        
        Arguments
        ---------
            anneal_fn : str
                Specification of anneal function type
            step : int
                Current step of VAE training cycle
            k : float
                KL annealing factor
            x0 : int
                Scalaing scalar for annealing

        Returns
        -------
            annealed k : float
                KL annealing factor
        """

        if anneal_fn == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_fn == 'linear':
            return min(1, step/x0)
        else:
            raise ValueError

    def reparameterise(self, hidden: Tensor, batch_size: int) -> Tensor:
        """
        Reparameterisation Trick (Kingma and Welling, 2014)

        Arguments
        ---------
            hidden : Tensor
                Hidden state of encoder
            batch_size : int
                Size of batch of sequences
        
        Returns
        -------
            z : Tensor
                SVAE latent space 
            mean : Tensor
                Gaussian mean
            logv : Tensor
                Gaussian variance
            std :  Tensor
                Gaussian standard deviation
        """
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.z_dim]))
        z = z * std + mean
        return z, mean, logv, std

    def loss_fn(self, logp: Tensor, target: Tensor, length: Tensor, mean: Tensor,
                logv: Tensor, anneal_fn: str, step: int, k: float, x0: int) -> Tensor:
        """
        Calculates SVAE loss in two parts: 1. Negative Log Likelihood, and 2. Kullback-Liebler divergence.
        As VAEs are self-supervised models, the input and targets are the same sequences.

        Arguments
        ---------
            logp : Tensor

            target : Tensor
                Batch of sequences (NOT TAGS)
            length : Tensor

            mean : Tensor

            logv : Tensor

            anneal_fm : str

            step : int

            k : float

            x0 : int

        Returns
        -------
            NLL_loss : TODO: type check
                Negative log likelihood
            KL_loss : TODO: type check
                Kullback-Liebler divergence
            KL_weight : TODO: type check
                TODO: review description
        """

        # Cut-off unnecessary padding from target and flatten
        # print(target)
        # target = trim_padded_seqs(batch_lengths=length,
        #                                 batch_sequences=target,
        #                                 pad_idx=self.pad_idx).view(-1)
        # print(target)
        
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        
        # Reshape logp tensor before calculating NLL
        logp = logp.view(-1, logp.size(2))

        # Negative log likelihood
        NLL_loss = self.NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = self.kl_anneal_fn(anneal_fn, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    def _encode(self, batch_sequences: Tensor) -> Tensor:
        """Forward pass through encoder RNN"""
        return self.encoder_rnn(batch_sequences)

    def _decode(self, batch_sequences: Tensor, hidden: Tensor) -> Tensor:
        """Forward pass through decoder RNN"""
        # print(f'shape of hidden size in _decode: {hidden.shape}')
        return self.decoder_rnn(batch_sequences, hidden)

    def inference(self, n=4, z=None):
        
        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.z_dim]))
        else:
            batch_size = z.size(0)
            
        hidden = self.z2hidden(z)   # latent2hidden...
        
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        
        hidden = hidden.unsqueeze(0)    # TODO check if correct
        
        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample
    
    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


class Discriminator(nn.Module):
    """ Adversarial Discriminator """
    def __init__(self, z_dim, fc_dim=128):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.fc_dim = fc_dim
        self.net = nn.Sequential(nn.Linear(self.z_dim, self.fc_dim),
                                 nn.ReLU(True),
                                 nn.Linear(self.fc_dim, self.fc_dim),
                                 nn.ReLU(True),
                                 nn.Linear(self.fc_dim, 1),
                                 nn.Sigmoid())
        # Initialise weights
        self.init_weights()

    def init_weights(self):
        """
        Initialises weights with Xavier method rather than Kaiming (TODO: investigate which is more suitable for LM and RNNs)
        - See: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html
        """
        for block in self._modules:
            for m in self._modules[block]:
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)
    
    
class Generator(nn.Module):
    """ Adversarial Generator """
    def __init__(self, z_dim, fc_dim=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc_dim = fc_dim
        self.net = nn.Sequential(nn.Linear(self.z_dim, self.fc_dim),
                                 nn.ReLU(True),
                                 nn.Linear(self.fc_dim, self.fc_dim),
                                 nn.ReLU(True),
                                 nn.Linear(self.fc_dim, 1),
                                 nn.Sigmoid())
        # Initialise weights
        self.init_weights()

    def init_weights(self):
        """
        Initialises weights with Xavier method rather than Kaiming (TODO: investigate which is more suitable for LM and RNNs)
        - See: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html
        """
        for block in self._modules:
            for m in self._modules[block]:
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)

if __name__ == '__main__':
    embeddings()