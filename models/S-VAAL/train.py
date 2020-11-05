"""
Trainer for generalisation of S-VAAL model.

@author: Tyler Bikaun
"""

# Imports
import yaml


from models import TaskLearner, SVAE, Discriminator
from utils import trim_padded_tags
# from data_generator import *


class Trainer:
    """ Prepares and trains S-VAAL model """
    def __init__(self, config):
        self.config = config


    def init_models(self):
        """ Initialises models, loss functions, optimisers and sets models to training mode """

        # Models
        self.task_learner = TaskLearner()
        self.svae = SVAE()
        self.discriminator = Discriminator()

        # Loss Functions


        # Optimisers


        # Training Modes


    def train(self):
        """ 
        Sequentially trains S-VAAL

        Training sequence
            ```
                for epoch in epochs:
                    train Task Learner
                    for step in steps:
                        train SVAE
                    for step in steps:
                        train Discriminator
            ```

        Arguments
        ---------

        Returns
        -------


        """
        




        





def main(config):
    # Initiate S-VAAL training

    Trainer(config)


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)