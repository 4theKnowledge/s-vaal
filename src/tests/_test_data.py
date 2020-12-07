"""
Tests for data.py
"""

import unittest

class Tests(unittest.TestCase):
    def setUp(self):
        # Init class
        self.sampler = Sampler(budget=10, sample_size=2)
        # Init random tensor
        self.data = torch.rand(size=(10,2,2))  # dim (batch, length, features)

        # output_classes = ['ORG', 'PER', 'LOC', 'MISC']

    # All sample tests are tested for:
    #   1. dims (_, length, features) for input and output Tensors
    #   2. batch size == sample size
    def test_sample_random(self):
        self.assertEqual(self.sampler.sample_random(self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_random(self.data).shape[0], self.sampler.sample_size)


def main():
    """ Runs basic functions for consistency and functionality checking """
    # tester = DataGenerator(config)

    # Generate sequences and their corresponding lengths
    # sequences, lengths = tester.build_sequences(no_sequences=2, max_sequence_length=10)
    
    # Generate output tags and build dataset with generated sequences, lengths and tags
    # tester.build_sequence_tags(sequences=sequences, lengths=lengths)
    
    # Generate vocabulary from sequences
    # tester.build_vocab(sequences)
    
    # Generate latents
    # tester.build_latents(no_sequences=2, z_dim=8)

    # Generate labelled/unlabelled datasets
    # dataset_l, dataset_u, vocab = tester.build_datasets(no_sequences=10, max_sequence_length=40, split=0.1)


    # Test dataset generator and dataloader
    # sequence_dataset = SequenceDataset(config, no_sequences=100, max_sequence_length=30, task_type="CLF")
    # dataloader = DataLoader(sequence_dataset, batch_size=7, shuffle=True, num_workers=0)

    # for i, batch in enumerate(dataloader):
    #     X, lens, y = batch
    #     print(i, X.shape, lens.shape, y.shape)

    pass

if __name__ == '__main__':
    unittest.main()