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