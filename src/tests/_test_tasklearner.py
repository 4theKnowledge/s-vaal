"""
Tests for tasklearner.py
"""

import unittest

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

if __name__ == '__main__':
    unittest.main()