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

if __name__ == '__main__':
    unittest.main()