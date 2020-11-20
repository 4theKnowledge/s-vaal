"""
Tests for models.py
"""

import unittest

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
        if self.model_type == 'svae':
            sequences, lengths = self.build_sequences(no_sequences=self.batch_size, max_sequence_length=self.max_sequence_length)
            self.dataset = self.build_sequence_tags(sequences=sequences, lengths=lengths)
            self.vocab = self.build_vocab(sequences)
            self.vocab_size = len(self.vocab)
        elif self.model_type == 'discriminator':
            self.dataset = self.build_latents(batch_size=self.batch_size, z_dim=self.z_dim)
    
    def init_model(self):
        """ Initialise neural network components including loss functions, optimisers and auxilliary functions """

        if self.model_type == 'discriminator':
            self.model = Discriminator(z_dim=self.z_dim).cuda()
            self.loss_fn = nn.BCELoss()
            self.optim = optim.Adam(self.model.parameters(), lr=self.config['Model']['Discriminator']['learning_rate'])
            self.model.train()

        elif self.model_type == 'svae':
            self.model = SVAE(config=self.config, vocab_size=self.vocab_size).cuda()
            # Note: loss_fn is accessed off of SVAE class rather that isntantiated here
            self.optim = optim.Adam(self.model.parameters(), lr=self.config['Model']['SVAE']['learning_rate'])
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
        if self.model_type == 'svae':
            step = 0    # used for SVAE KL-annealing
            for epoch in range(self.epochs):
                for batch_sequences, batch_lengths, batch_tags in self.dataset:

                    if torch.cuda.is_available():
                        batch_sequences = batch_sequences.cuda()
                        batch_lengths = batch_lengths.cuda()
                        batch_tags = batch_tags.cuda()

                    if epoch == 0:
                        print(f'Shapes | Sequences: {batch_sequences.shape} Lengths: {batch_lengths.shape} Tags: {batch_tags.shape}')

                    batch_size = batch_sequences.size(0)

                    self.model.zero_grad()
                    # Strip off tag padding (similar to variable length sequences via pack padded methods)
                    batch_tags = trim_padded_seqs(batch_lengths=batch_lengths,
                                                batch_sequences=batch_tags,
                                                pad_idx=self.pad_idx).view(-1)

                    # Forward pass through model
                    logp, mean, logv, z = self.model(batch_sequences, batch_lengths)
                    # print(f'logp: {logv} mean: {mean} logv: {logv} z: {z}')
                    
                    # Calculate loss and backpropagate error through model
                    # print(logp.shape)
                    # print(logp)
                    # print(batch_sequences.shape)
                    # print(batch_sequences)
                    NLL_loss, KL_loss, KL_weight = self.model.loss_fn(logp=logp, target=batch_sequences,
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


if __name__ == '__main__':
    unittest.main()