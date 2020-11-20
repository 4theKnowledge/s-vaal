"""
Module for connections including No-SQL and configuration files
"""

import yaml

def load_config():
    """ Loads configuration file from disk
    """
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        print(e)


class Mongo:
    def __init__(self):
        pass

    def write_experiment(self):
        pass