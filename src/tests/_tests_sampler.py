class Tests(unittest.TestCase):
    
    def _sim_model(self, data: Tensor) -> Tensor:
        """ Simulated model for generating uncertainity scores. Intention
            is to be a placeholder until real models are used and for testing."""
        return torch.rand(size=(data.shape[0],))
    
    def setUp(self):
        # Init class
        self.sampler = Sampler(budget=10)
        # Init random tensor
        self.data = torch.rand(size=(10,2,2))  # dim (batch, length, features)
        # Params
        self.budget = 18

    # All sample tests are tested for:
    #   1. dims (_, length, features) for input and output Tensors
    #   2. batch size == sample size
    def test_sample_random(self):
        self.assertEqual(self.sampler.sample_random(self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_random(self.data).shape[0], self.sampler.budget)

    def test_sample_least_confidence(self):
        self.assertEqual(self.sampler.sample_least_confidence(model=self.sampler._sim_model, data=self.data).shape[1:], self.data.shape[1:])
        self.assertEqual(self.sampler.sample_least_confidence(model=self.sampler._sim_model, data=self.data).shape[0], self.sampler.budget)

    # def test_sample_bayesian(self):
    #     self.assertEqual(self.sampler.sample_bayesian(model=self.sampler._sim_model, no_models=3, data=self.data).shape[1:], self.data.shape[1:])
    #     self.assertEqual(self.sampler.sample_bayesian(model=self.sampler._sim_model, no_models=3, data=self.data).shape[0], self.sampler.budget)

    # def test_adversarial_sample(self):
        # self.assertEqual(self.sampler.sample_adversarial(self.data).shape[1:], self.data.shape[1:])
        # self.assertEqual(self.sampler.sample_adversarial(self.data).shape[0], self.sampler.budget)