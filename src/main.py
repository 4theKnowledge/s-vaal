"""
Main script which orchestrates model initialisation, model training, etc.

@author: Tyler Bikaun
"""

import yaml


class SomeClass:
    """ """
    def __init__(self):
        pass


def main(config):
    """
    Does something someday...
    """
    # will do something one day
    pass

if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    # Seeds
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed'])

    main(config)