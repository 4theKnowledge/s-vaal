"""
Contains model initialisation procedures and test functionality

@author: Tyler Bikaun
"""

# Imports
import yaml
import numpy as np

from data_generator import DataGenerator
from utils import to_var, trim_padded_tags

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
Tensor = torch.Tensor


class TaskLearner(nn.Module):
    """ Initialises a sequence based task learner (RNN based) 
    
    Arguments
    ---------
        embedding_dim : int

        hidden_dim : int

        vocab_size : int

        tagset_size :
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super(TaskLearner, self).__init__()

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # TODO: Implement padding_idx=self.pad_idx

        # Current sequence tagger is an LSTM (TODO: implement more advanced sequence taggers and options)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer that maps hidden state space from LSTM to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

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
        # Sort and pack padded sequence for variable length LSTM
        sorted_lengths, sorted_idx = torch.sort(batch_lengths, descending=True)
        batch_sequences = batch_sequences[sorted_idx]
        input_embeddings = self.word_embeddings(batch_sequences)
        packed_input = rnn_utils.pack_padded_sequence(input_embeddings, sorted_lengths.data.tolist(), batch_first=True)
        
        lstm_out, _ = self.lstm(packed_input)
        
        # Unpack padded sequence
        padded_outputs = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        
        # Project into tag space
        tag_space = self.hidden2tag(padded_outputs.view(-1, padded_outputs.size(2)))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class SVAE(nn.Module):
    """ Sentence Variational Autoencoder"""
    def __init__(self, config, vocab_size: int):
        super(SVAE, self).__init__()
        utils_config = config['Utils']
        svae_config = config['Model']['SVAE']
        svae_config_parameters = svae_config['Parameters']

        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        # This will need to be cleaned up rather than specified explicity
        # TODO: make more general when code working e.g. use <PAD> and link to word2idx
        self.max_sequence_length = config['Model']['max_sequence_length']
        self.pad_idx = utils_config['special_tokens']['pad_idx']
        self.eos_idx = vocab_size + utils_config['special_tokens']['eos_idx']
        self.sos_idx = vocab_size + utils_config['special_tokens']['sos_idx']
        self.unk_idx = vocab_size + utils_config['special_tokens']['unk_idx']
        
        self.vocab_size = vocab_size + len(utils_config['special_tokens'])
                
        # RNN settings
        self.rnn_type = svae_config_parameters['rnn_type']
        self.bidirectional = svae_config_parameters['bidirectional']
        self.num_layers = svae_config_parameters['num_layers']
        self.hidden_size = svae_config_parameters['hidden_size']
        self.embedding_size = svae_config_parameters['embedding_size']

        # Latent space dimension
        self.z_dim = svae_config_parameters['latent_size']
        
        # Embedding initialisation
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.word_dropout_rate = svae_config_parameters['word_dropout']
        self.embedding_dropout = nn.Dropout(p=svae_config_parameters['embedding_dropout'])
        
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
        
        # Initialise encoder-decoder RNNs
        # Note: these models are identical in structure
        self.encoder_rnn = rnn(input_size=self.embedding_size,
                               hidden_size=self.hidden_size, 
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(input_size=self.embedding_size,
                               hidden_size=self.hidden_size, 
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        # Hidden factor is used for expanding dimensionality if bidirectionality and multi-layer functionality is used
        self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers
        
        # Initialisation of FC layers
        # These map from the encoder to the latent space
        self.hidden2mean = nn.Linear(self.hidden_size * self.hidden_factor, self.z_dim)
        self.hidden2logv = nn.Linear(self.hidden_size * self.hidden_factor, self.z_dim)
        self.z2hidden = nn.Linear(self.z_dim, self.hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.vocab_size)
        
        # Initialise partial loss function
        self.NLL = nn.NLLLoss(ignore_index=self.pad_idx,
                                reduction='sum')   # TODO: REVIEW args to better understand


        # # SVAE loss function
        # # # TODO: Review args to ensure they're correct
        # self.svae_NLL = nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')

    def forward(self, input_sequence: Tensor, length: Tensor) -> Tensor:
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
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)   # trick for packed padding
        input_sequence = input_sequence[sorted_idx]
        
        # ENCODER
        input_embedding = self.embedding(input_sequence)
#         print(input_embedding.shape)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self._encode(packed_input)
        
        if self.bidirectional or 1 < self.num_layers:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            # .squeeze() -> Returns a tensor with all the dimensions of input of size 1 removed.
            # print(f'hidden shape before squeeze {hidden.shape}')
#             hidden = hidden.squeeze()   # doesn't work? gives wrong dimension down stream... must be due to their data format or bidirection/n_layers? TODO: test.
            # print(f'hidden shape after squeeze {hidden.shape}')
            pass

        
        # Reparameterisation trick!
        z, mean, logv, std = self.reparameterise(hidden, batch_size)
        
        # DECODER
        if 0 < self.word_dropout_rate:
            prob = torch.rand(input_sequence.size())

            if torch.cuda.is_available():
                prob = prob.cuda()

            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1

            decoder_input_sequence = input_sequence.clone()
            
#             print(vocab_size)
#             print(self.unk_idx)
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx

#             print(decoder_input_sequence)
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
#         print(f'logp before view {logp.shape}\n')
#         print(f'b {b} s {s} no emb {self.embedding.num_embeddings}')
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def kl_anneal_fn(self, anneal_fn: str, step: int, k: float, x0: int):
        """
        TODO: add description from Bowman et al., 2016
        
        Arguments
        ---------
            anneal_fn : str
                Specification of anneal function type
            step : int
                TODO: figure out
            k : float
                TODO: figure out
            x0 : TODO: figure out
                TODO: figure out

        Returns
        -------
            annealed k : float
                TODO: figure out
        
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
                Hidden state of encoder RNN
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
        
        Arguments
        ---------
            logp : Tensor

            target : Tensor

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

        # Cut-off unnecessary padding from target and flatten is not required herein as doing this before loss_function
        # print(target)
        # print(target.shape)
        # print(logp.shape)
        # print(logp.view(-1, logp.size(2)).shape)

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
        return self.decoder_rnn(batch_sequences, hidden)

    def inference(self):
        """"""
        pass

    def _sample(self):
        """"""
        pass
    
    def _save_sample(self):
        """"""
        pass


class Discriminator(nn.Module):
    """ Adversarial Discriminator
    
    Arguments
    ---------
        z_dim : int
            Dimension of latent space
    """
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()

        self.z_dim = z_dim  #config['Model']['Discriminator']['z_dim']  # dimension of latent space

        self.net = nn.Sequential(
                                nn.Linear(self.z_dim, 128),
                                nn.ReLU(True),
                                nn.Linear(128,128),
                                nn.ReLU(True),
                                nn.Linear(128,1),
                                nn.Sigmoid()
                                )

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
        return self.net(z.type(torch.FloatTensor))


class Tester(DataGenerator):
    """ Tests individual training routines for each of the three neural models
    Note: Testing is only on a single batch of generated sequences, rather than the true S-VAAL training method.
    
    Arguments
    ---------
        config : yaml
            Configuration file for model initialisation and testing 
    """
    def __init__(self, config):
        DataGenerator.__init__(self, config)   # Allows access properties and build methods
        self.config = config
        
        # Testing data properties
        self.batch_size = config['Tester']['batch_size']
        self.max_sequence_length = config['Tester']['max_sequence_length']
        self.z_dim = 8  # Discriminator latent space size
        self.embedding_dim = 128
        
        # Test run properties
        self.model_type = config['Tester']['model_type'].lower()
        self.epochs = config['Tester']['epochs']
        self.iterations = config['Tester']['iterations']

        # Exe
        self.training_routine()
        
    def init_data(self):
        """ Initialise synthetic sequence data for testing """
        if self.model_type in ['task_learner', 'svae']:
            sequences, lengths = self.build_sequences(batch_size=self.batch_size, max_sequence_length=self.max_sequence_length)
            self.dataset = self.build_sequence_tags(sequences=sequences, lengths=lengths)
            self.vocab = self.build_vocab(sequences)
            self.vocab_size = len(self.vocab)
        elif self.model_type == 'discriminator':
            self.dataset = self.build_latents(batch_size=self.batch_size, z_dim=self.z_dim)
    
    def init_model(self):
        """ Initialise neural network components including loss functions, optimisers and auxilliary functions """

        if self.model_type == 'task_learner':
            self.model = TaskLearner(embedding_dim=self.embedding_dim, hidden_dim=128, vocab_size=self.vocab_size, tagset_size=self.tag_space_size)   # self.tag_space_size from DataGenerator
            # print(self.self.model)
            self.loss_fn = nn.NLLLoss()
            self.optim = optim.SGD(self.model.parameters(), lr=0.1)
            # Set model to train mode
            self.model.train()

        elif self.model_type == 'discriminator':
            self.model = Discriminator(z_dim=self.z_dim)
            self.loss_fn = nn.BCELoss()
            self.optim = optim.Adam(self.model.parameters(), lr=0.001)
            self.model.train()

        elif self.model_type == 'svae':
            self.model = SVAE(config=self.config, vocab_size=self.vocab_size)
            # Note: loss_fn is accessed off of SVAE class rather that isntantiated here
            self.optim = optim.Adam(self.model.parameters(), lr=0.001)
            self.model.train()

        else:
            raise ValueError

    def training_routine(self):
        """ Abstract training routine """
        # Initialise training data and model for testing
        print(f'TRAINING {self.model_type.upper()}')
        self.init_data()
        self.init_model()

        # Train model
        if self.model_type in ['task_learner', 'svae']:
            step = 0    # used for SVAE KL-annealing
            for epoch in range(self.epochs):
                for batch_sequences, batch_lengths, batch_tags in self.dataset:
                    if epoch == 0:
                        print(f'Shapes | Sequences: {batch_sequences.shape} Lengths: {batch_lengths.shape} Tags: {batch_tags.shape}')

                    batch_size = batch_sequences.size(0)

                    self.model.zero_grad()
                    # Strip off tag padding (similar to variable length sequences via pack padded methods)
                    batch_tags = trim_padded_tags(batch_lengths=batch_lengths,
                                                batch_tags=batch_tags,
                                                pad_idx=self.pad_idx).view(-1)

                    # Task learner has different routine than SVAE
                    if self.model_type == 'task_learner':
                        # Forward pass through model
                        tag_scores = self.model(batch_sequences, batch_lengths)
                        
                        # Calculate loss and backpropagate error through model
                        loss = self.loss_fn(tag_scores, batch_tags)
                        loss.backward()
                        self.optim.step()
                    
                    # SVAE training routine
                    else:
                        # Forward pass through model
                        logp, mean, logv, z = self.model(batch_sequences, batch_lengths)
                        # print(f'logp: {logv} mean: {mean} logv: {logv} z: {z}')
                        
                        # Calculate loss and backpropagate error through model
                        NLL_loss, KL_loss, KL_weight = self.model.loss_fn(logp=logp, target=batch_tags,
                                                                            length=batch_lengths, mean=mean,
                                                                            logv=logv, anneal_fn='logistic',
                                                                            step=step, k=0.0025, x0=2500)

                        loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                        loss.backward()
                        self.optim.step()
                    
                    print(f'Epoch: {epoch} - Loss: {loss.data.detach():0.2f}')
                    step += 1

        elif self.model_type == 'discriminator':
            # The discriminator takes in different arguments than the task learner and SVAE so it must be trained differently
            # TODO: Add unlabelled and labelled functionality here rather than just one...
            # In this instance, dataset is a batch of data, but TODO will include making this a generator function to yield from.
            for i in range(self.iterations):
                preds = self.model(self.dataset)
                real_labels = torch.ones(preds.size(0))
                loss = self.loss_fn(preds, real_labels)

                print(f'Iteration: {i} - Loss {loss.data.detach():0.2f}')
    
        else:
            raise ValueError


def main(config):
    """
    Initialises models, generates synthetic sequence data and runs tests on each model
    to ensure they're working correctly.
    """
    # Generate synthetic data
    # data_generator = DataGenerator(config)
    # sequences, lengths = data_generator.build_sequences(batch_size=2, max_sequence_length=10)
    # test_dataset = data_generator.build_sequence_tags(sequences=sequences, lengths=lengths)
    # vocab = data_generator.build_vocab(sequences)

    # Initialise models
    # pass

    # Run tests
    Tester(config)




if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)