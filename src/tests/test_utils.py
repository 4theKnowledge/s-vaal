"""
Utility functions for performing tests on components. Within this module includes functions
for generating synthetic sequence data, ..., ...

Each class is similar to those in the real model.

@author: Tyler Bikaun
"""

class Trainer():
    """
    """
    def __init__(self):
        pass

    def _init_dataset_gen(self):
        """ Initialises dataset for model training """
        # Currently will be using generated data, but in the future will be real.

        self.train_dataset_l = SequenceDataset(self.config, no_sequences=8, max_sequence_length=self.max_sequence_length, task_type=self.task_type)
        self.train_dataloader_l = DataLoader(self.train_dataset_l, batch_size=2, shuffle=True, num_workers=0)

        self.train_dataset_u = SequenceDataset(self.config, no_sequences=16, max_sequence_length=self.max_sequence_length, task_type=self.task_type)
        self.train_dataloader_u = DataLoader(self.train_dataset_u, batch_size=2, shuffle=True, num_workers=0)

        # Concatenate sequences in X_l and X_u to build vocabulary for downstream
        self.vocab = self.build_vocab(sequences = torch.cat((self.train_dataset_l.sequences, self.train_dataset_u.sequences)))
        self.vocab_size = len(self.vocab)

        print('---- DATA SUCCESSFULLY INITIALISED ----')
        
        
class Tests(unittest.TestCase):
    def setUp(self):
        self.tensor_shape = (100,10,20)
        self.sequences = torch.stack([torch.randint(0,10,size=(10,)) for _ in range(self.tensor_shape[0])])
        self.split_2 = (0.2,0.8)
        self.split_3 = (0.1,0.1,0.8)
        self.rand_tensor = torch.randint(0,10,size=self.tensor_shape)

    def test_data_split(self):
        ds1, ds2 = split_data(dataset=self.rand_tensor, splits=self.split_2)
        self.assertEqual(len(ds1), self.tensor_shape[0]*self.split_2[0])
        self.assertEqual(len(ds2), self.tensor_shape[0]*self.split_2[1])
        ds1, ds2, ds3 = split_data(dataset=self.rand_tensor, splits=self.split_3)
        self.assertEqual(len(ds1), self.tensor_shape[0]*self.split_3[0])
        self.assertEqual(len(ds2), self.tensor_shape[0]*self.split_3[1])
        self.assertEqual(len(ds3), self.tensor_shape[0]*self.split_3[2])

    def test_get_lengths(self):
        self.assertEqual(len(get_lengths(self.sequences)), self.tensor_shape[0])