"""
Contains model initialisation information and procedures

@author: Tyler Bikaun
"""

# Imports
import yaml

from data_generator import DataGenerator

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
Tensor = torch.Tensor


class TaskLearner(nn.Module):
    """ Sequence based task learner """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(TaskLearner, self).__init__()

        # Word Embeddings (TODO: Implement pre-trained word embeddings)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # TODO: Implement padding_idx=self.pad_idx

        # Current sequence tagger is an LSTM (TODO: implement more advanced sequence taggers and options)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer that maps hidden state space from LSTM to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)


    def forward(self, batch_sequences: Tensor, batch_lengths: Tensor) -> Tensor:
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
    def __init__(self):
        pass


class Discriminator(nn.Module):
    """ """
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

    def forward(self, z):
        return self.net(z.type(torch.FloatTensor))


class Tester(DataGenerator):
    def __init__(self, config):
        DataGenerator.__init__(self, config)   # Allows access properties and build methods
        
        # Testing data properties
        self.batch_size = config['Tester']['batch_size']
        self.max_sequence_length = config['Tester']['max_sequence_length']
        self.z_dim = 8  # Discriminator latent space size
        
        # Test run properties
        self.model_type = config['Tester']['model_type'].lower()
        self.epochs = config['Tester']['epochs']
        self.iterations = config['Tester']['iterations']

        # Exe
        self.training_routine()
        
    def init_data(self):
        """ Initialise synthetic sequence data for testing """
        if self.model_type in ['task_learner', 'SVAE']:
            sequences, lengths = self.build_sequences(batch_size=self.batch_size, max_sequence_length=self.max_sequence_length)
            self.dataset = self.build_sequence_tags(sequences=sequences, lengths=lengths)
            self.vocab = self.build_vocab(sequences)
            self.vocab_size = len(self.vocab)
        elif self.model_type == 'discriminator':
            self.dataset = self.build_latents(batch_size=self.batch_size, z_dim=self.z_dim)

    def init_model(self):
        """ Initialise neural network components including loss functions, optimisers and auxilliary functions """

        if self.model_type == 'task_learner':
            self.model = TaskLearner(embedding_dim=128, hidden_dim=128, vocab_size=self.vocab_size, tagset_size=self.tag_space_size)   # self.tag_space_size from DataGenerator
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
            pass

        else:
            raise ValueError

    def training_routine(self):
        """ Abstract training routine """
        # Initialise training data and model for testing
        print(f'TRAINING {self.model_type.upper()}')
        self.init_data()
        self.init_model()

        # Train model
        if self.model_type in ['task_learner', 'SVAE']:
            for epoch in range(self.epochs):
                for batch_sequences, batch_lengths, batch_tags in self.dataset:
                    if epoch == 0:
                        print(f'Shapes | Sequences: {batch_sequences.shape} Lengths: {batch_lengths.shape} Tags: {batch_tags.shape}')

                    self.model.zero_grad()

                    # Get max length of longest sequence in batch so it can be used to filter tags
                    sorted_lengths, _ = torch.sort(batch_lengths, descending=True)   # longest seq at index 0
                    longest_seq = sorted_lengths[0].data.numpy()
                    longest_seq_len = longest_seq[longest_seq != 0][0]   # remove padding (TODO: change to pad_idx in the future)
                    
                    # Get predictions from model
                    tag_scores = self.model(batch_sequences, batch_lengths)
                    
                    # Strip off as much padding as possible similar to (variable length sequences via pack padded methods)
                    batch_tags = torch.stack([tags[:longest_seq_len] for tags in batch_tags])
                    batch_tags = batch_tags.view(-1)
                    
                    # Calculate loss and backpropigate error through model
                    loss = self.loss_fn(tag_scores, batch_tags)
                    loss.backward()
                    self.optim.step()
                    
                    print(f'Epoch: {epoch} - Loss: {loss.data.detach():0.2f}')

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